package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/cobra"
)

// ReviewSession represents an active review session
type ReviewSession struct {
	SessionID   string    `json:"session_id"` // unique session identifier
	ProjectPath string    `json:"project_path"`
	IndexPath   string    `json:"index_path"` // full path to the review index
	StartedAt   time.Time `json:"started_at"`
}

// generateSessionID creates a unique session identifier
func generateSessionID() string {
	// timestamp + random hash for uniqueness
	h := sha256.Sum256([]byte(fmt.Sprintf("%d", time.Now().UnixNano())))
	return hex.EncodeToString(h[:])[:12]
}

// getReviewIndexName generates a unique index name for this session
func getReviewIndexName(sessionID string, projectPath string) string {
	base := filepath.Base(projectPath)
	return fmt.Sprintf("review_%s_%s", base, sessionID)
}

// isOllamaRunning checks if ollama server is responding
func isOllamaRunning() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://localhost:11434/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// startOllama starts ollama serve in background
func startOllama() error {
	if isOllamaRunning() {
		return nil
	}

	fmt.Println("starting ollama...")
	cmd := exec.Command("ollama", "serve")
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start ollama: %w", err)
	}

	// wait for ollama to be ready
	for i := 0; i < 30; i++ {
		time.Sleep(500 * time.Millisecond)
		if isOllamaRunning() {
			fmt.Println("ollama ready")
			return nil
		}
	}

	return fmt.Errorf("ollama failed to start within 15 seconds")
}

// ensureEmbeddingModel ensures the embedding model is available
func ensureEmbeddingModel(model string) error {
	fmt.Printf("checking embedding model: %s\n", model)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, "ollama", "pull", model)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd.Run()
}

// getReviewSessionPath returns the path to the review session file
func getReviewSessionPath() (string, error) {
	configDir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	sessionDir := filepath.Join(configDir, "lr")
	if err := os.MkdirAll(sessionDir, 0755); err != nil {
		return "", err
	}
	return filepath.Join(sessionDir, "review_session.json"), nil
}

// getReviewIndexDir returns the path for review indexes (separate from regular indexes)
func getReviewIndexDir() (string, error) {
	dataDir, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	reviewDir := filepath.Join(dataDir, ".local", "share", "lr", "review")
	if err := os.MkdirAll(reviewDir, 0755); err != nil {
		return "", err
	}
	return reviewDir, nil
}

// runReviewStart starts a review session
func runReviewStart(_ *cobra.Command, _ []string) error {
	// check if there's already an active session
	existingSession, err := loadReviewSession()
	if err == nil {
		// check if the index file still exists (session might be stale from crash)
		if _, statErr := os.Stat(existingSession.IndexPath); os.IsNotExist(statErr) {
			// stale session - clean it up
			fmt.Printf("cleaning up stale session (index missing): %s\n", existingSession.SessionID)
			_ = clearReviewSession()
		} else {
			return fmt.Errorf("review session already active for: %s\nrun 'lr review stop' first", existingSession.ProjectPath)
		}
	}

	// get current directory
	projectPath, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %w", err)
	}

	fmt.Printf("starting review session for: %s\n\n", projectPath)

	// start ollama if not running
	if err := startOllama(); err != nil {
		return err
	}

	// ensure embedding model is available
	embModel := "nomic-embed-text"
	if err := ensureEmbeddingModel(embModel); err != nil {
		return fmt.Errorf("failed to pull embedding model: %w", err)
	}

	// create ollama client for indexing
	ollamaClient := NewOllamaClient(embModel)

	// generate unique session ID and index path
	sessionID := generateSessionID()
	reviewDir, err := getReviewIndexDir()
	if err != nil {
		return err
	}
	indexName := getReviewIndexName(sessionID, projectPath)
	indexPath := filepath.Join(reviewDir, indexName+".lrindex")

	// load files (code + docs)
	extensions := []string{".go", ".js", ".ts", ".jsx", ".tsx", ".templ", ".md"}
	fmt.Printf("scanning files...\n")
	loadResult, err := LoadFilesByExtensionsWithStatsAndSplit(projectPath, extensions, "mixed", 100*1024, false, true)
	if err != nil {
		return fmt.Errorf("failed to load files: %w", err)
	}

	fmt.Printf("found %d files to index\n", len(loadResult.Documents))
	if len(loadResult.SkippedFiles) > 0 {
		fmt.Printf("skipped %d files\n", len(loadResult.SkippedFiles))
	}

	// chunk documents
	fmt.Println("chunking files...")
	var chunks []Chunk
	for _, doc := range loadResult.Documents {
		docChunks := ChunkDocument(doc, 1000)
		chunks = append(chunks, docChunks...)
	}
	fmt.Printf("created %d chunks\n", len(chunks))

	// create embeddings
	fmt.Println("generating embeddings with ollama...")
	store := NewVectorStore()
	store.Metadata.SourcePath = projectPath
	store.Metadata.ReviewIndex = true
	store.Metadata.EmbeddingModel = embModel

	for i, chunk := range chunks {
		embedding, err := ollamaClient.GetEmbedding(chunk.Text)
		if err != nil {
			return fmt.Errorf("failed to get embedding for chunk %d: %w", i, err)
		}

		store.Add(chunk, embedding)

		// progress indicator
		if (i+1)%10 == 0 || i == len(chunks)-1 {
			fmt.Printf("\r  embedded %d/%d chunks", i+1, len(chunks))
		}
	}
	fmt.Println()

	// set metadata
	store.Metadata.IndexedAt = time.Now().Format(time.RFC3339)
	store.Metadata.ChunkCount = len(chunks)
	store.Metadata.FileCount = len(loadResult.Documents)

	// save index
	if err := store.Save(indexPath); err != nil {
		return fmt.Errorf("failed to save index: %w", err)
	}

	// save session info
	session := ReviewSession{
		SessionID:   sessionID,
		ProjectPath: projectPath,
		IndexPath:   indexPath,
		StartedAt:   time.Now(),
	}
	if err := saveReviewSession(&session); err != nil {
		return fmt.Errorf("failed to save session: %w", err)
	}

	fmt.Printf("\nreview session started!\n")
	fmt.Printf("  session: %s\n", sessionID)
	fmt.Printf("  index: %s\n", indexPath)
	fmt.Printf("  chunks: %d\n", len(chunks))
	fmt.Println("\nwatching for changes... (Ctrl+C to stop)")

	// start watching - this blocks until interrupted
	return startWatching(&session, store, indexPath, ollamaClient)
}

// runReviewStop stops the review session
func runReviewStop(_ *cobra.Command, _ []string) error {
	session, err := loadReviewSession()
	if err != nil {
		return fmt.Errorf("no active review session: %w", err)
	}

	// delete the index using the stored path
	if err := os.Remove(session.IndexPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete index: %w", err)
	}

	// clear session
	if err := clearReviewSession(); err != nil {
		return fmt.Errorf("failed to clear session: %w", err)
	}

	fmt.Printf("review session stopped (session %s)\n", session.SessionID)
	fmt.Printf("  deleted: %s\n", session.IndexPath)

	return nil
}

// runReviewStatus shows the current review session status
func runReviewStatus(_ *cobra.Command, _ []string) error {
	session, err := loadReviewSession()
	if err != nil {
		fmt.Println("no active review session")
		return nil
	}

	fmt.Printf("active review session:\n")
	fmt.Printf("  session: %s\n", session.SessionID)
	fmt.Printf("  project: %s\n", session.ProjectPath)
	fmt.Printf("  index: %s\n", session.IndexPath)
	fmt.Printf("  started: %s\n", session.StartedAt.Format(time.RFC3339))
	fmt.Printf("  duration: %s\n", time.Since(session.StartedAt).Round(time.Second))

	// check if ollama is running
	if isOllamaRunning() {
		fmt.Println("  ollama: running")
	} else {
		fmt.Println("  ollama: not running")
	}

	return nil
}

// saveReviewSession saves the session to disk
func saveReviewSession(session *ReviewSession) error {
	sessionPath, err := getReviewSessionPath()
	if err != nil {
		return err
	}

	data, err := json.Marshal(session)
	if err != nil {
		return err
	}

	return os.WriteFile(sessionPath, data, 0644)
}

// loadReviewSession loads the session from disk
func loadReviewSession() (*ReviewSession, error) {
	sessionPath, err := getReviewSessionPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(sessionPath)
	if err != nil {
		return nil, err
	}

	var session ReviewSession
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, err
	}

	return &session, nil
}

// clearReviewSession removes the session file
func clearReviewSession() error {
	sessionPath, err := getReviewSessionPath()
	if err != nil {
		return err
	}

	return os.Remove(sessionPath)
}

// runReviewWatch starts watching for file changes and updates the index (standalone command)
func runReviewWatch(_ *cobra.Command, _ []string) error {
	session, err := loadReviewSession()
	if err != nil {
		return fmt.Errorf("no active review session. run 'lr review start' first")
	}

	// ensure ollama is running
	if !isOllamaRunning() {
		if err := startOllama(); err != nil {
			return err
		}
	}

	// load existing index using stored path
	store := NewVectorStore()
	if err := store.Load(session.IndexPath); err != nil {
		return fmt.Errorf("failed to load index: %w", err)
	}

	// create ollama client
	ollamaClient := NewOllamaClient("nomic-embed-text")

	fmt.Println("watching for changes... (Ctrl+C to stop)")
	return startWatching(session, store, session.IndexPath, ollamaClient)
}

// startWatching is the shared watch loop used by both start and watch commands
func startWatching(session *ReviewSession, store *VectorStore, indexPath string, ollamaClient *OllamaClient) error {
	// create watcher
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("failed to create watcher: %w", err)
	}
	defer watcher.Close()

	// add directories recursively
	watchedDirs := 0
	err = filepath.Walk(session.ProjectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip errors
		}
		if info.IsDir() {
			// skip common non-code directories
			base := filepath.Base(path)
			if base == "node_modules" || base == ".git" || base == "vendor" ||
				base == "dist" || base == "build" || base == ".next" {
				return filepath.SkipDir
			}
			if err := watcher.Add(path); err == nil {
				watchedDirs++
			}
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("failed to walk directory: %w", err)
	}

	fmt.Printf("watching %d directories for changes...\n", watchedDirs)

	// track extensions we care about
	watchedExts := map[string]bool{
		".go": true, ".js": true, ".ts": true, ".jsx": true,
		".tsx": true, ".templ": true, ".md": true,
	}

	// debounce changes (collect changes over 500ms before processing)
	pendingChanges := make(map[string]bool)
	var debounceTimer *time.Timer

	processChanges := func() {
		if len(pendingChanges) == 0 {
			return
		}

		// copy and clear pending
		files := make([]string, 0, len(pendingChanges))
		for f := range pendingChanges {
			files = append(files, f)
		}
		pendingChanges = make(map[string]bool)

		fmt.Printf("\nupdating %d file(s)...\n", len(files))

		for _, filePath := range files {
			// check if file still exists
			info, err := os.Stat(filePath)
			if err != nil {
				// file deleted - remove from index
				relPath, _ := filepath.Rel(session.ProjectPath, filePath)
				removed := store.RemoveBySource([]string{relPath})
				if removed > 0 {
					fmt.Printf("  removed %d chunks from deleted file: %s\n", removed, filepath.Base(filePath))
				}
				continue
			}

			// skip if too large
			if info.Size() > 100*1024 {
				continue
			}

			// read file content
			content, err := os.ReadFile(filePath)
			if err != nil {
				continue
			}

			// create document and chunk
			relPath, _ := filepath.Rel(session.ProjectPath, filePath)

			// remove old chunks for this file
			store.RemoveBySource([]string{relPath})
			doc := Document{
				Content:  string(content),
				Source:   relPath,
				Metadata: map[string]string{"type": "code"},
			}

			chunks := ChunkDocument(doc, 1000)
			if len(chunks) == 0 {
				continue
			}

			// generate embeddings for new chunks
			for _, chunk := range chunks {
				embedding, err := ollamaClient.GetEmbedding(chunk.Text)
				if err != nil {
					fmt.Printf("  error embedding %s: %v\n", filepath.Base(filePath), err)
					continue
				}
				store.Add(chunk, embedding)
			}

			fmt.Printf("  updated: %s (%d chunks)\n", filepath.Base(filePath), len(chunks))
		}

		// save updated index
		store.Metadata.IndexedAt = time.Now().Format(time.RFC3339)
		store.Metadata.ChunkCount = len(store.Chunks)
		// update file count based on unique sources
		uniqueFiles := make(map[string]bool)
		for _, chunk := range store.Chunks {
			uniqueFiles[chunk.Source] = true
		}
		store.Metadata.FileCount = len(uniqueFiles)
		if err := store.Save(indexPath); err != nil {
			fmt.Printf("  error saving index: %v\n", err)
		}
	}

	// handle signals for graceful shutdown (Ctrl+C, Ctrl+Z, kill)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM, syscall.SIGTSTP)

	for {
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				return nil
			}

			// only care about write/create events
			if event.Op&(fsnotify.Write|fsnotify.Create) == 0 {
				continue
			}

			// check extension
			ext := strings.ToLower(filepath.Ext(event.Name))
			if !watchedExts[ext] {
				continue
			}

			// skip excluded files
			if ShouldExcludeFile(event.Name) {
				continue
			}

			// add to pending changes
			pendingChanges[event.Name] = true

			// reset debounce timer
			if debounceTimer != nil {
				debounceTimer.Stop()
			}
			debounceTimer = time.AfterFunc(500*time.Millisecond, processChanges)

		case err, ok := <-watcher.Errors:
			if !ok {
				return nil
			}
			fmt.Printf("watcher error: %v\n", err)

		case <-sigChan:
			fmt.Println("\nstopping review session...")
			if debounceTimer != nil {
				debounceTimer.Stop()
				processChanges() // process any pending changes
			}
			// clean up: delete index and clear session
			if err := os.Remove(indexPath); err != nil && !os.IsNotExist(err) {
				fmt.Printf("warning: failed to delete index: %v\n", err)
			}
			if err := clearReviewSession(); err != nil {
				fmt.Printf("warning: failed to clear session: %v\n", err)
			}
			fmt.Printf("session stopped, index deleted\n")
			return nil
		}
	}
}
