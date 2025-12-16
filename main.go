package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
)

const (
	maxChunkSize       = 1500
	checkpointInterval = 100 // save every 100 chunks
)

var (
	// index command flags
	srcPath      string
	useCode      bool
	useDocs      bool
	outPath      string
	outName      string
	dryRun       bool
	maxFileSize  int64
	splitLarge   bool
	includeTests bool
	updateIndex  bool
	useGit       bool

	// query command flags
	topK         int
	querySources []string
	useMCP       bool
	noSynthesize bool

	// mcp command flags
	noPreload bool
	reloadPid int
	reloadAll bool

	// model configuration flags
	chatModel      string
	embeddingModel string
)

// model aliases for convenience
var chatModelAliases = map[string]string{
	"sonnet":      "claude-sonnet-4-5-20250929",
	"haiku":       "claude-haiku-4-5-20251001",
	"opus":        "claude-opus-4-5-20251101",
	"gpt-4o":      "gpt-4o",
	"gpt-4o-mini": "gpt-4o-mini",
}

var embeddingModelAliases = map[string]string{
	"openai":  "text-embedding-3-small",
	"voyage":  "voyage-code-2",
	"voyage3": "voyage-3",
	"ollama":  "nomic-embed-text",
}

// default chat model
const defaultChatModel = "claude-sonnet-4-5-20250929"

// resolveChatModel resolves a model alias to its full model ID
func resolveChatModel(model string) string {
	if model == "" {
		return defaultChatModel
	}
	if resolved, ok := chatModelAliases[model]; ok {
		return resolved
	}
	return model // assume it's a full model ID
}

// resolveEmbeddingModel resolves an embedding model alias to its full model ID
func resolveEmbeddingModel(model string) string {
	if model == "" {
		return "" // will be auto-detected
	}
	if resolved, ok := embeddingModelAliases[model]; ok {
		return resolved
	}
	return model // assume it's a full model ID
}

var rootCmd = &cobra.Command{
	Use:   "lr",
	Short: "LocalRag - local-first RAG system for code and documentation",
	Long:  `LocalRag indexes and queries code repositories and documentation using local vector storage.`,
}

var indexCmd = &cobra.Command{
	Use:   "index",
	Short: "Index a repository or directory",
	Long:  `Index code and/or documentation from a local directory or git repository.`,
	RunE:  runIndex,
}

var queryCmd = &cobra.Command{
	Use:   "query [question]",
	Short: "Query indexed repositories",
	Long:  `Ask a question and get answers from indexed repositories.`,
	Args:  cobra.MinimumNArgs(1),
	RunE:  runQuery,
}

var interactiveCmd = &cobra.Command{
	Use:   "interactive",
	Short: "Start interactive query mode",
	Long:  `Start an interactive session to ask multiple questions.`,
	RunE:  runInteractive,
}

var mcpCmd = &cobra.Command{
	Use:   "mcp",
	Short: "Start MCP server for Claude Code integration",
	Long:  `Start a Model Context Protocol server on stdio for integration with Claude Code.`,
	RunE:  runMCP,
}

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List all available vector store indexes",
	Long:  `Display all available vector store indexes with their metadata.`,
	RunE:  runList,
}

var setupCmd = &cobra.Command{
	Use:   "setup",
	Short: "Print MCP configuration for Claude Code integration",
	Long:  `Print the MCP server configuration to add to Claude Code's config file.`,
	RunE:  runSetup,
}

var pathsCmd = &cobra.Command{
	Use:   "paths",
	Short: "Show where lr stores data and config",
	Long:  `Display the directories used for storing indexes and configuration.`,
	Run:   runPaths,
}

var updateAllCmd = &cobra.Command{
	Use:   "update-all",
	Short: "Update all indexes that have source paths",
	Long:  `Incrementally update all indexes that have recorded source paths. Creates a backup before updating.`,
	RunE:  runUpdateAll,
}

var reviewCmd = &cobra.Command{
	Use:   "review",
	Short: "Code review context using local ollama embeddings",
	Long:  `Start/stop a review session that indexes your project locally for code review context.`,
}

var reviewStartCmd = &cobra.Command{
	Use:   "start",
	Short: "Start a review session (indexes current directory with ollama)",
	Long: `Start a review session. This will:
1. Start ollama if not running
2. Pull the embedding model if needed
3. Index the current directory
4. Enable watch mode for live updates`,
	RunE: runReviewStart,
}

var reviewStopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop the review session and delete the index",
	RunE:  runReviewStop,
}

var reviewStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show current review session status",
	RunE:  runReviewStatus,
}

var reviewWatchCmd = &cobra.Command{
	Use:   "watch",
	Short: "Watch for file changes and update the index in real-time",
	Long:  `Start watching for file changes. When a file is saved, it will be re-indexed automatically.`,
	RunE:  runReviewWatch,
}

func init() {
	// load .env file if it exists (check current dir, then config dir)
	envPath := getEnvFilePath()
	if err := LoadEnv(envPath); err != nil {
		// silently ignore if .env doesn't exist
		if !os.IsNotExist(err) {
			fmt.Printf("warning: error loading .env file: %v\n", err)
		}
	}

	// index command flags
	indexCmd.Flags().StringVar(&srcPath, "src", "", "source directory or URL to index (required)")
	indexCmd.Flags().BoolVar(&useCode, "code", true, "index code files (.go, .js, .ts, etc) [default: true]")
	indexCmd.Flags().BoolVar(&useDocs, "docs", true, "index documentation files (.md) [default: true]")
	indexCmd.Flags().StringVar(&outPath, "out", "", "exact output path (e.g., indexes/myindex.lrindex)")
	indexCmd.Flags().StringVar(&outName, "out-name", "", "output name (saved as indexes/{name}_YYYYMMDD.lrindex)")
	indexCmd.Flags().BoolVar(&dryRun, "dry-run", false, "show what would be indexed without actually indexing")
	indexCmd.Flags().Int64Var(&maxFileSize, "max-file-size", 100*1024, "maximum file size in bytes (default 100KB)")
	indexCmd.Flags().BoolVar(&splitLarge, "split-large", false, "split large files into sections instead of skipping them")
	indexCmd.Flags().BoolVar(&includeTests, "include-tests", true, "include test files (useful usage examples) [default: true]")
	indexCmd.Flags().BoolVar(&updateIndex, "update", false, "incrementally update existing index (only re-index changed files)")
	indexCmd.Flags().BoolVar(&useGit, "git", false, "use git to detect changes (default: file mtime)")
	indexCmd.MarkFlagRequired("src")

	// query command flags
	queryCmd.Flags().IntVar(&topK, "top-k", 3, "number of relevant chunks to retrieve")
	queryCmd.Flags().StringSliceVar(&querySources, "sources", []string{}, "filter by source names (comma-separated, e.g., nats-server,docs)")
	queryCmd.Flags().BoolVar(&useMCP, "use-mcp", false, "use running MCP server instead of loading indexes directly")
	queryCmd.Flags().BoolVar(&noSynthesize, "no-synthesize", false, "return raw chunks without LLM synthesis (only works with --use-mcp)")

	// mcp command flags
	mcpCmd.Flags().BoolVar(&noPreload, "no-preload", false, "disable vector store preloading (allows on-the-fly updates)")
	mcpCmd.Flags().IntVar(&reloadPid, "reload", 0, "send reload signal to mcp server with given pid")
	mcpCmd.Flags().BoolVar(&reloadAll, "reload-all", false, "send reload signal to all lr mcp processes")

	// model configuration flags (persistent, available to all commands)
	rootCmd.PersistentFlags().StringVar(&chatModel, "model", "", "chat model to use (aliases: sonnet, haiku, opus, gpt-4o, gpt-4o-mini)")
	rootCmd.PersistentFlags().StringVar(&embeddingModel, "embedding-model", "", "embedding model (aliases: openai, voyage, voyage3, ollama)")

	// update-all command flags
	updateAllCmd.Flags().BoolVar(&useGit, "git", false, "use git to detect changes (default: file mtime)")

	// add commands
	rootCmd.AddCommand(indexCmd)
	rootCmd.AddCommand(queryCmd)
	rootCmd.AddCommand(interactiveCmd)
	rootCmd.AddCommand(mcpCmd)
	rootCmd.AddCommand(listCmd)
	rootCmd.AddCommand(setupCmd)
	rootCmd.AddCommand(pathsCmd)
	rootCmd.AddCommand(updateAllCmd)

	// review command with subcommands
	reviewCmd.AddCommand(reviewStartCmd)
	reviewCmd.AddCommand(reviewStopCmd)
	reviewCmd.AddCommand(reviewStatusCmd)
	reviewCmd.AddCommand(reviewWatchCmd)
	rootCmd.AddCommand(reviewCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func getLLMClient() (LLMClient, error) {
	openaiKey := os.Getenv("OPENAI_API_KEY")
	claudeKey := os.Getenv("ANTHROPIC_API_KEY")
	voyageKey := os.Getenv("VOYAGE_API_KEY")

	// resolve model aliases
	resolvedChatModel := resolveChatModel(chatModel)
	resolvedEmbeddingModel := resolveEmbeddingModel(embeddingModel)

	// ollama: local embeddings (no api key needed, just needs ollama running)
	if embeddingModel == "ollama" || resolvedEmbeddingModel == "nomic-embed-text" {
		embModel := resolvedEmbeddingModel
		if embModel == "" {
			embModel = "nomic-embed-text"
		}
		fmt.Printf("using ollama embeddings (%s) + claude chat (%s)\n", embModel, resolvedChatModel)
		return NewOllamaClaudeClient(embModel, resolvedChatModel), nil
	}

	// priority order for embedding+chat combinations
	if voyageKey != "" && claudeKey != "" {
		embModel := resolvedEmbeddingModel
		if embModel == "" {
			embModel = "voyage-code-2"
		}
		fmt.Printf("using voyage ai embeddings (%s) + claude chat (%s)\n", embModel, resolvedChatModel)
		return NewVoyageClaudeClient(voyageKey, claudeKey, embModel, resolvedChatModel), nil
	} else if openaiKey != "" && claudeKey != "" {
		embModel := resolvedEmbeddingModel
		if embModel == "" {
			embModel = "text-embedding-3-small"
		}
		fmt.Printf("using openai embeddings (%s) + claude chat (%s)\n", embModel, resolvedChatModel)
		return NewHybridClient(openaiKey, claudeKey, embModel, resolvedChatModel), nil
	} else if openaiKey != "" {
		embModel := resolvedEmbeddingModel
		if embModel == "" {
			embModel = "text-embedding-3-small"
		}
		// for openai-only, use gpt model if no claude model specified
		chatModelToUse := resolvedChatModel
		if chatModel == "" {
			chatModelToUse = "gpt-4o-mini"
		}
		fmt.Printf("using openai for embeddings (%s) and chat (%s)\n", embModel, chatModelToUse)
		return NewOpenAIClient(openaiKey, chatModelToUse, embModel), nil
	}

	return nil, fmt.Errorf("no api key found. please set one of:\n" +
		"  - OPENAI_API_KEY (for openai only)\n" +
		"  - OPENAI_API_KEY + ANTHROPIC_API_KEY (hybrid mode)\n" +
		"  - VOYAGE_API_KEY + ANTHROPIC_API_KEY (recommended for code!)\n" +
		"  - --embedding-model=ollama (local embeddings, no api key needed)")
}

func estimateCost(numChunks int) {
	openaiKey := os.Getenv("OPENAI_API_KEY")
	claudeKey := os.Getenv("ANTHROPIC_API_KEY")
	voyageKey := os.Getenv("VOYAGE_API_KEY")

	// average chunk size is around 1000 characters = ~250 tokens
	avgTokensPerChunk := 250
	totalTokens := numChunks * avgTokensPerChunk

	// pricing as of january 2025 (per 1M tokens)
	const (
		openaiEmbeddingCost = 0.020 // text-embedding-3-small: $0.020 / 1M tokens
		voyageEmbeddingCost = 0.120 // voyage-code-2: $0.120 / 1M tokens
	)

	var embeddingCost float64
	var provider string

	// determine which provider will be used
	if voyageKey != "" && claudeKey != "" {
		embeddingCost = voyageEmbeddingCost
		provider = "voyage ai"
	} else if openaiKey != "" {
		embeddingCost = openaiEmbeddingCost
		provider = "openai"
	} else {
		fmt.Println("Estimated cost: unable to determine (no api keys configured)")
		return
	}

	// calculate cost
	cost := (float64(totalTokens) / 1_000_000.0) * embeddingCost

	fmt.Printf("Estimated cost: $%.4f (%s embeddings)\n", cost, provider)
	fmt.Printf("  - %d chunks × %d tokens/chunk = %d tokens\n", numChunks, avgTokensPerChunk, totalTokens)
	fmt.Printf("  - %s: $%.3f per 1M tokens\n", provider, embeddingCost)
}

func runIndex(_ *cobra.Command, _ []string) error {
	// validate flags
	if !dryRun {
		if outPath == "" && outName == "" {
			return fmt.Errorf("either --out or --out-name is required when not using --dry-run")
		}
		if outPath != "" && outName != "" {
			return fmt.Errorf("cannot specify both --out and --out-name")
		}
	}

	// --update requires --out-name (to find existing index)
	if updateIndex && outName == "" {
		return fmt.Errorf("--update requires --out-name to find existing index")
	}

	// --git requires --update
	if useGit && !updateIndex {
		return fmt.Errorf("--git only works with --update")
	}

	// construct final output path
	var finalOutPath string
	if outName != "" {
		timestamp := time.Now().Format("20060102")
		indexDir := getDefaultIndexDir()
		finalOutPath = filepath.Join(indexDir, fmt.Sprintf("%s_%s.lrindex", outName, timestamp))
	} else {
		finalOutPath = outPath
	}

	// handle incremental update
	if updateIndex {
		return runIncrementalIndex(finalOutPath)
	}

	fmt.Printf("analyzing source: %s\n", srcPath)

	// check if source exists
	if _, err := os.Stat(srcPath); os.IsNotExist(err) {
		return fmt.Errorf("source directory not found: %s", srcPath)
	}

	// determine which extensions to load
	var extensions []string
	var docType string
	if useCode && useDocs {
		extensions = []string{".go", ".js", ".ts", ".jsx", ".tsx", ".templ", ".md"}
		docType = "mixed"
	} else if useDocs {
		extensions = []string{".md"}
		docType = "markdown"
	} else {
		extensions = []string{".go", ".js", ".ts", ".jsx", ".tsx", ".templ"}
		docType = "code"
	}

	// load files with statistics
	fmt.Printf("scanning files from %s...\n", srcPath)
	loadResult, err := LoadFilesByExtensionsWithStatsAndSplit(srcPath, extensions, docType, maxFileSize, splitLarge, includeTests)
	if err != nil {
		return fmt.Errorf("failed to load files: %w", err)
	}

	fmt.Printf("\n=== SCAN RESULTS ===\n")
	fmt.Printf("Total files found: %d\n", loadResult.TotalFiles)
	fmt.Printf("Files to index: %d\n", len(loadResult.Documents))
	fmt.Printf("Files skipped: %d\n", len(loadResult.SkippedFiles))

	if len(loadResult.SkippedFiles) > 0 {
		fmt.Println("\nSkipped files:")
		for _, sf := range loadResult.SkippedFiles {
			fmt.Printf("  - %s (%s)\n", sf.Path, sf.Reason)
		}
	}

	// chunk documents
	fmt.Println("\nchunking files...")
	var chunks []Chunk
	for _, doc := range loadResult.Documents {
		docChunks := ChunkDocument(doc, maxChunkSize)
		chunks = append(chunks, docChunks...)
	}
	fmt.Printf("created %d chunks\n", len(chunks))

	// if dry run, just show summary and exit
	if dryRun {
		fmt.Println("\n=== DRY RUN SUMMARY ===")
		fmt.Printf("Would index %d files into %d chunks\n", len(loadResult.Documents), len(chunks))
		fmt.Printf("Estimated embeddings to generate: %d\n", len(chunks))

		// estimate cost based on available api keys
		estimateCost(len(chunks))

		fmt.Printf("Estimated time: ~%d minutes\n", (len(chunks)*50)/1000/60)
		return nil
	}

	// proceed with actual indexing
	llm, err := getLLMClient()
	if err != nil {
		return err
	}

	// create simple loader that returns already loaded docs
	loader := func(dir string) ([]Document, error) {
		return loadResult.Documents, nil
	}

	fmt.Printf("\nindexing source: %s\n", srcPath)
	if err := indexSingleSource(llm, srcPath, finalOutPath, loader); err != nil {
		return fmt.Errorf("error indexing source: %w", err)
	}
	fmt.Println("indexing complete!")
	return nil
}

func runQuery(_ *cobra.Command, args []string) error {
	question := strings.Join(args, " ")

	// if --use-mcp flag is set, query via MCP server
	if useMCP {
		if len(querySources) > 0 {
			return fmt.Errorf("--sources flag is not supported with --use-mcp (use MCP server configuration)")
		}

		synthesize := !noSynthesize
		result, err := queryViaMCP(question, topK, synthesize)
		if err != nil {
			return fmt.Errorf("error querying via MCP: %w", err)
		}

		fmt.Println(result)
		return nil
	}

	// standard query mode (load indexes directly)
	llm, err := getLLMClient()
	if err != nil {
		return err
	}

	// load vector stores
	indexDir := getDefaultIndexDir()
	mss := NewMultiSourceStore(indexDir)

	// if specific sources requested, load only those
	if len(querySources) > 0 {
		for _, source := range querySources {
			if err := mss.LoadSource(source); err != nil {
				return fmt.Errorf("error loading source %s: %w", source, err)
			}
		}
	} else {
		// otherwise load all
		if err := mss.LoadAll(); err != nil {
			return fmt.Errorf("error loading vector stores: %w\nrun 'lr index' to index repositories first", err)
		}
	}

	if len(mss.Sources) == 0 {
		return fmt.Errorf("no vector stores found\nrun 'lr index' to index repositories first")
	}

	fmt.Printf("loaded %d sources: %v\n", len(mss.Sources), mss.ListSources())

	rag := NewRAGMultiSource(mss, llm)

	answer, results, err := rag.QueryWithSources(question, topK, querySources)
	if err != nil {
		return fmt.Errorf("error querying: %w", err)
	}

	printResults(question, answer, results)
	return nil
}

func runList(_ *cobra.Command, _ []string) error {
	indexDir := getDefaultIndexDir()

	// check if directory exists
	if _, err := os.Stat(indexDir); os.IsNotExist(err) {
		fmt.Println("no vector stores found")
		fmt.Println("run 'lr index' to create your first index")
		return nil
	}

	// find all index files (.lrindex or .json for backward compat)
	patterns := []string{
		filepath.Join(indexDir, "*.lrindex"),
		filepath.Join(indexDir, "*.json"),
	}
	var files []string
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return fmt.Errorf("error searching for indexes: %w", err)
		}
		files = append(files, matches...)
	}

	// filter out checkpoint files
	var validFiles []string
	for _, file := range files {
		if !strings.Contains(filepath.Base(file), "checkpoint") {
			validFiles = append(validFiles, file)
		}
	}

	if len(validFiles) == 0 {
		fmt.Println("no vector stores found")
		fmt.Println("run 'lr index' to create your first index")
		return nil
	}

	fmt.Printf("found %d vector store(s):\n\n", len(validFiles))

	// load each vector store and display metadata
	for _, file := range validFiles {
		vs := NewVectorStore()
		if err := vs.Load(file); err != nil {
			fmt.Printf("  ✗ %s (error loading: %v)\n", filepath.Base(file), err)
			continue
		}

		baseName := filepath.Base(file)
		sourceName := strings.TrimSuffix(baseName, ".json")

		// strip nats_ prefix if present
		if strings.HasPrefix(sourceName, "nats_") {
			sourceName = sourceName[5:]
		}

		// strip timestamp suffix if present
		if parts := strings.Split(sourceName, "_"); len(parts) > 1 {
			lastPart := parts[len(parts)-1]
			if len(lastPart) == 8 {
				sourceName = strings.Join(parts[:len(parts)-1], "_")
			}
		}

		fmt.Printf("  • %s\n", sourceName)
		fmt.Printf("    file: %s\n", baseName)
		fmt.Printf("    chunks: %d\n", len(vs.Chunks))
		if vs.Metadata.FileCount > 0 {
			fmt.Printf("    files indexed: %d\n", vs.Metadata.FileCount)
		}
		if vs.Metadata.SourcePath != "" {
			fmt.Printf("    source: %s\n", vs.Metadata.SourcePath)
		}
		if vs.Metadata.IndexedAt != "" {
			fmt.Printf("    indexed: %s\n", vs.Metadata.IndexedAt)
		}
		fmt.Println()
	}

	return nil
}

func runUpdateAll(_ *cobra.Command, _ []string) error {
	indexDir := getDefaultIndexDir()

	// check if directory exists
	if _, err := os.Stat(indexDir); os.IsNotExist(err) {
		return fmt.Errorf("no indexes found - run 'lr index' first")
	}

	// find all index files
	pattern := filepath.Join(indexDir, "*.lrindex")
	files, err := filepath.Glob(pattern)
	if err != nil {
		return fmt.Errorf("error searching for indexes: %w", err)
	}

	// filter out checkpoint and temp files
	var validFiles []string
	for _, file := range files {
		base := filepath.Base(file)
		if !strings.Contains(base, "checkpoint") && !strings.Contains(base, ".tmp.") {
			validFiles = append(validFiles, file)
		}
	}

	if len(validFiles) == 0 {
		return fmt.Errorf("no indexes found")
	}

	// find indexes that have source paths and scan for changes
	type indexInfo struct {
		path        string
		name        string
		sourcePath  string
		isGitRepo   bool
		vs          *VectorStore
		changeSet   *ChangeSet
		needsPull   bool
		behindCount int
	}
	var updatable []indexInfo

	fmt.Println("scanning indexes for changes...")
	for _, file := range validFiles {
		vs := NewVectorStore()
		if err := vs.Load(file); err != nil {
			fmt.Printf("  ✗ %s: error loading\n", filepath.Base(file))
			continue
		}

		if vs.Metadata.SourcePath == "" {
			fmt.Printf("  - %s: no source path\n", filepath.Base(file))
			continue
		}

		// check if source path exists
		if _, err := os.Stat(vs.Metadata.SourcePath); os.IsNotExist(err) {
			fmt.Printf("  ✗ %s: source not found: %s\n", filepath.Base(file), vs.Metadata.SourcePath)
			continue
		}

		// extract name from filename
		name := strings.TrimSuffix(filepath.Base(file), ".lrindex")
		// remove date suffix
		if parts := strings.Split(name, "_"); len(parts) > 1 {
			lastPart := parts[len(parts)-1]
			if len(lastPart) == 8 {
				name = strings.Join(parts[:len(parts)-1], "_")
			}
		}

		info := indexInfo{
			path:       file,
			name:       name,
			sourcePath: vs.Metadata.SourcePath,
			isGitRepo:  isGitRepo(vs.Metadata.SourcePath),
			vs:         vs,
		}

		// determine extensions (default to code)
		extensions := []string{".go", ".js", ".ts", ".jsx", ".tsx", ".templ"}

		// detect changes
		if info.isGitRepo && vs.Metadata.LastCommit != "" {
			// check if behind remote
			behind := getGitBehindCount(vs.Metadata.SourcePath)
			if behind > 0 {
				info.needsPull = true
				info.behindCount = behind
			}

			// git-based change detection
			cs, err := detectChangesGit(vs.Metadata.SourcePath, vs.Metadata.LastCommit, extensions)
			if err == nil {
				info.changeSet = cs
			}
		} else if vs.Metadata.IndexedAt != "" {
			// mtime-based change detection
			indexedAt, err := time.Parse(time.RFC3339, vs.Metadata.IndexedAt)
			if err == nil {
				cs, err := detectChangesMtime(vs.Metadata.SourcePath, indexedAt, vs.Metadata.IndexedFiles, extensions)
				if err == nil {
					info.changeSet = cs
				}
			}
		}

		updatable = append(updatable, info)
	}

	if len(updatable) == 0 {
		fmt.Println("\nno updatable indexes found (indexes need source paths)")
		return nil
	}

	// show scan results
	fmt.Printf("\n=== SCAN RESULTS ===\n")
	var totalChanges int
	var needsWork []indexInfo
	var pullWarnings []string

	for _, idx := range updatable {
		if idx.needsPull {
			pullWarnings = append(pullWarnings, fmt.Sprintf("  ⚠ %s: %d commits behind remote (consider: cd %s && git pull)",
				idx.name, idx.behindCount, idx.sourcePath))
		}

		if idx.changeSet != nil && idx.changeSet.HasChanges() {
			changes := len(idx.changeSet.Added) + len(idx.changeSet.Modified) + len(idx.changeSet.Deleted)
			totalChanges += changes
			needsWork = append(needsWork, idx)
			fmt.Printf("  ✓ %s: %d added, %d modified, %d deleted\n",
				idx.name, len(idx.changeSet.Added), len(idx.changeSet.Modified), len(idx.changeSet.Deleted))
		} else if idx.changeSet != nil {
			fmt.Printf("  - %s: up to date\n", idx.name)
		} else {
			fmt.Printf("  ? %s: could not detect changes\n", idx.name)
		}
	}

	// show pull warnings
	if len(pullWarnings) > 0 {
		fmt.Printf("\n=== GIT WARNINGS ===\n")
		for _, w := range pullWarnings {
			fmt.Println(w)
		}
	}

	// if no work needed, exit early
	if len(needsWork) == 0 {
		fmt.Println("\nall indexes are up to date - nothing to do")
		return nil
	}

	fmt.Printf("\n%d index(es) need updating with %d total file changes\n", len(needsWork), totalChanges)

	// create backup directory
	backupDir := filepath.Join(indexDir, fmt.Sprintf("backup_%s", time.Now().Format("20060102_150405")))
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return fmt.Errorf("failed to create backup directory: %w", err)
	}
	fmt.Printf("\ncreating backup in %s...\n", filepath.Base(backupDir))

	// backup all index files
	for _, file := range validFiles {
		src := file
		dst := filepath.Join(backupDir, filepath.Base(file))
		srcFile, err := os.Open(src)
		if err != nil {
			return fmt.Errorf("failed to open %s for backup: %w", filepath.Base(src), err)
		}
		dstFile, err := os.Create(dst)
		if err != nil {
			srcFile.Close()
			return fmt.Errorf("failed to create backup file %s: %w", filepath.Base(dst), err)
		}
		if _, err := io.Copy(dstFile, srcFile); err != nil {
			srcFile.Close()
			dstFile.Close()
			return fmt.Errorf("failed to backup %s: %w", filepath.Base(src), err)
		}
		srcFile.Close()
		dstFile.Close()
	}
	fmt.Printf("backed up %d index files\n", len(validFiles))

	// get LLM client
	llm, err := getLLMClient()
	if err != nil {
		return fmt.Errorf("failed to initialize LLM client: %w", err)
	}

	// update only indexes that need work
	fmt.Println("\nupdating indexes...")
	var successCount, failCount int

	for _, idx := range needsWork {
		fmt.Printf("\n=== Updating %s ===\n", idx.name)

		// set global variables for runIncrementalIndex
		srcPath = idx.sourcePath
		outName = idx.name

		// determine output path
		finalOutPath := filepath.Join(indexDir, fmt.Sprintf("%s_%s.lrindex", idx.name, time.Now().Format("20060102")))

		// run incremental update using existing function
		if err := runIncrementalIndexWithLLM(llm, finalOutPath); err != nil {
			fmt.Printf("✗ failed to update %s: %v\n", idx.name, err)
			failCount++
			continue
		}

		successCount++
	}

	fmt.Printf("\n=== Summary ===\n")
	fmt.Printf("updated: %d\n", successCount)
	fmt.Printf("failed: %d\n", failCount)
	fmt.Printf("backup: %s\n", backupDir)

	return nil
}

func runInteractive(_ *cobra.Command, _ []string) error {
	llm, err := getLLMClient()
	if err != nil {
		return err
	}

	// load all vector stores
	indexDir := getDefaultIndexDir()
	mss := NewMultiSourceStore(indexDir)
	if err := mss.LoadAll(); err != nil {
		return fmt.Errorf("error loading vector stores: %w\nrun 'lr index' to index repositories first", err)
	}

	if len(mss.Sources) == 0 {
		return fmt.Errorf("no vector stores found\nrun 'lr index' to index repositories first")
	}

	fmt.Printf("loaded %d sources: %v\n", len(mss.Sources), mss.ListSources())

	rag := NewRAGMultiSource(mss, llm)

	fmt.Println("=== localrag interactive mode ===")
	fmt.Println("ask questions about your indexed repositories. type 'exit' to quit.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("question: ")
		if !scanner.Scan() {
			break
		}

		question := strings.TrimSpace(scanner.Text())
		if question == "" {
			continue
		}

		if strings.ToLower(question) == "exit" || strings.ToLower(question) == "quit" {
			fmt.Println("goodbye!")
			break
		}

		// query the rag system
		answer, results, err := rag.Query(question, topK)
		if err != nil {
			fmt.Printf("error: %v\n\n", err)
			continue
		}

		printResults(question, answer, results)
	}

	return nil
}

func runMCP(_ *cobra.Command, _ []string) error {
	return serveMCP()
}

func runPaths(_ *cobra.Command, _ []string) {
	fmt.Println("=== lr data directories ===")
	fmt.Println()
	fmt.Printf("indexes:  %s\n", getDefaultIndexDir())
	fmt.Printf("config:   %s\n", getConfigDir())
	fmt.Printf("env file: %s\n", getEnvFilePath())
	fmt.Println()
	fmt.Println("these directories follow the XDG base directory specification")
	fmt.Println("you can override them with environment variables:")
	fmt.Println("  XDG_DATA_HOME   - base directory for data files")
	fmt.Println("  XDG_CONFIG_HOME - base directory for config files")
}

func runSetup(_ *cobra.Command, _ []string) error {
	// get the absolute path to the lr binary
	execPath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("failed to determine executable path: %w", err)
	}

	// resolve symlinks to get the actual binary path
	realPath, err := filepath.EvalSymlinks(execPath)
	if err != nil {
		realPath = execPath
	}

	fmt.Println("=== claude code mcp setup ===")
	fmt.Println()
	fmt.Println("run this command to register lr with claude code:")
	fmt.Println()
	fmt.Printf("  claude mcp add --transport stdio lr -- %s mcp\n", realPath)
	fmt.Println()
	fmt.Println("after running this command:")
	fmt.Println("  - restart claude code to activate the mcp server")
	fmt.Println("  - ask questions about your indexed repositories naturally")
	fmt.Println()
	fmt.Println("notes:")
	fmt.Println("  - the mcp server preloads indexes at startup for fast queries")
	fmt.Println("  - to pick up newly indexed repositories, restart claude code")
	fmt.Println("  - use 'lr paths' to see where your indexes are stored")
	fmt.Println()

	return nil
}

func indexSingleSource(llm LLMClient, srcPath, outPath string, loader func(string) ([]Document, error)) error {
	start := time.Now()

	// check if source exists
	if _, err := os.Stat(srcPath); os.IsNotExist(err) {
		return fmt.Errorf("source directory not found: %s", srcPath)
	}

	// load files
	fmt.Printf("loading files from %s...\n", srcPath)
	docs, err := loader(srcPath)
	if err != nil {
		return fmt.Errorf("failed to load files: %w", err)
	}
	fmt.Printf("loaded %d files\n", len(docs))

	// chunk documents
	fmt.Println("chunking files...")
	var chunks []Chunk
	for _, doc := range docs {
		docChunks := ChunkDocument(doc, maxChunkSize)
		chunks = append(chunks, docChunks...)
	}
	fmt.Printf("created %d chunks\n", len(chunks))

	// use the output path as-is (timestamp already applied in runIndex if using --out-name)
	outputFile := outPath

	// ensure output directory exists
	outputDir := filepath.Dir(outputFile)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// checkpoint file (same name but with .checkpoint before extension)
	var checkpointFile string
	if strings.HasSuffix(outputFile, ".lrindex") {
		checkpointFile = strings.Replace(outputFile, ".lrindex", ".checkpoint.lrindex", 1)
	} else {
		checkpointFile = strings.Replace(outputFile, ".json", ".checkpoint.json", 1)
	}

	// try to load checkpoint if it exists
	vs := NewVectorStore()
	startIdx := 0

	if _, err := os.Stat(checkpointFile); err == nil {
		fmt.Printf("found checkpoint, resuming...\n")
		if err := vs.Load(checkpointFile); err != nil {
			fmt.Printf("warning: could not load checkpoint: %v\n", err)
		} else {
			startIdx = len(vs.Chunks)
			fmt.Printf("resuming from chunk %d/%d\n", startIdx, len(chunks))
		}
	}

	// create embeddings
	var bar *progressbar.ProgressBar
	if startIdx == 0 {
		bar = progressbar.NewOptions(len(chunks),
			progressbar.OptionSetDescription("generating embeddings"),
			progressbar.OptionShowCount(),
			progressbar.OptionSetWidth(40),
			progressbar.OptionThrottle(100*time.Millisecond),
			progressbar.OptionShowIts(),
			progressbar.OptionSetItsString("chunks"),
		)
	} else {
		remaining := len(chunks) - startIdx
		bar = progressbar.NewOptions(remaining,
			progressbar.OptionSetDescription("resuming embeddings"),
			progressbar.OptionShowCount(),
			progressbar.OptionSetWidth(40),
			progressbar.OptionThrottle(100*time.Millisecond),
			progressbar.OptionShowIts(),
			progressbar.OptionSetItsString("chunks"),
		)
	}

	for i := startIdx; i < len(chunks); i++ {
		chunk := chunks[i]
		embedding, err := llm.GetEmbedding(chunk.Text)
		if err != nil {
			return fmt.Errorf("failed to get embedding for chunk %d (size: %d chars, ~%d tokens): %w",
				i, len(chunk.Text), len(chunk.Text)/4, err)
		}

		vs.Add(chunk, embedding)
		bar.Add(1)

		// save checkpoint periodically
		if (i+1)%checkpointInterval == 0 {
			if err := vs.Save(checkpointFile); err != nil {
				fmt.Printf("\nwarning: failed to save checkpoint: %v\n", err)
			}
		}

		// small delay to avoid rate limits
		time.Sleep(50 * time.Millisecond)
	}
	bar.Finish()
	fmt.Println()

	// set metadata before saving
	absPath, _ := filepath.Abs(srcPath)
	vs.Metadata.SourcePath = absPath
	vs.Metadata.IndexedAt = time.Now().Format(time.RFC3339)
	vs.Metadata.ChunkCount = len(vs.Chunks)
	vs.Metadata.FileCount = len(docs)

	// populate indexed files list
	fileSet := make(map[string]bool)
	for _, doc := range docs {
		fileSet[doc.Source] = true
	}
	vs.Metadata.IndexedFiles = make([]string, 0, len(fileSet))
	for f := range fileSet {
		vs.Metadata.IndexedFiles = append(vs.Metadata.IndexedFiles, f)
	}

	// record git commit if in a git repo
	if isGitRepo(srcPath) {
		if commit, err := getGitHeadCommit(srcPath); err == nil {
			vs.Metadata.LastCommit = commit
		}
	}

	// save final vector store
	fmt.Printf("saving %s...\n", outputFile)
	if err := vs.Save(outputFile); err != nil {
		return fmt.Errorf("failed to save vector store: %w", err)
	}

	// remove checkpoint file since we completed successfully
	if _, err := os.Stat(checkpointFile); err == nil {
		os.Remove(checkpointFile)
	}

	elapsed := time.Since(start)
	fmt.Printf("✓ indexed successfully (%d chunks in %s)\n", len(chunks), elapsed.Round(time.Second))
	return nil
}

func runIncrementalIndex(finalOutPath string) error {
	// get LLM client
	llm, err := getLLMClient()
	if err != nil {
		return err
	}
	return runIncrementalIndexWithLLM(llm, finalOutPath)
}

func runIncrementalIndexWithLLM(llm LLMClient, finalOutPath string) error {
	start := time.Now()

	// find existing index
	indexDir := getDefaultIndexDir()
	existingIndex, err := findExistingIndex(indexDir, outName)
	if err != nil {
		return fmt.Errorf("cannot update: %w", err)
	}
	fmt.Printf("found existing index: %s\n", filepath.Base(existingIndex))

	// load existing index
	vs := NewVectorStore()
	if err := vs.Load(existingIndex); err != nil {
		return fmt.Errorf("failed to load existing index: %w", err)
	}
	fmt.Printf("loaded %d existing chunks\n", len(vs.Chunks))

	// migrate old indexes: populate IndexedFiles from chunk sources if empty
	if len(vs.Metadata.IndexedFiles) == 0 && len(vs.Chunks) > 0 {
		fileSet := make(map[string]bool)
		for _, chunk := range vs.Chunks {
			fileSet[chunk.Source] = true
		}
		vs.Metadata.IndexedFiles = make([]string, 0, len(fileSet))
		for f := range fileSet {
			vs.Metadata.IndexedFiles = append(vs.Metadata.IndexedFiles, f)
		}
		fmt.Printf("migrated index: found %d indexed files from chunks\n", len(vs.Metadata.IndexedFiles))
	}

	// check source exists
	if _, err := os.Stat(srcPath); os.IsNotExist(err) {
		return fmt.Errorf("source directory not found: %s", srcPath)
	}

	// determine extensions
	var extensions []string
	var docType string
	if useCode && useDocs {
		extensions = []string{".go", ".js", ".ts", ".jsx", ".tsx", ".templ", ".md"}
		docType = "mixed"
	} else if useDocs {
		extensions = []string{".md"}
		docType = "markdown"
	} else {
		extensions = []string{".go", ".js", ".ts", ".jsx", ".tsx", ".templ"}
		docType = "code"
	}

	// detect changes - auto-use git if index has LastCommit and source is a git repo
	var changeSet *ChangeSet
	canUseGit := vs.Metadata.LastCommit != "" && isGitRepo(srcPath)
	if useGit || canUseGit {
		// git-based detection
		if !isGitRepo(srcPath) {
			return fmt.Errorf("--git specified but %s is not a git repository", srcPath)
		}
		if vs.Metadata.LastCommit == "" {
			return fmt.Errorf("existing index has no LastCommit - full re-index required")
		}
		fmt.Printf("detecting changes since commit %s...\n", vs.Metadata.LastCommit[:8])
		changeSet, err = detectChangesGit(srcPath, vs.Metadata.LastCommit, extensions)
		if err != nil {
			return fmt.Errorf("git change detection failed: %w", err)
		}
	} else {
		// mtime-based detection
		var indexedAt time.Time
		if vs.Metadata.IndexedAt != "" {
			indexedAt, err = time.Parse(time.RFC3339, vs.Metadata.IndexedAt)
			if err != nil {
				return fmt.Errorf("cannot parse IndexedAt timestamp: %w", err)
			}
		} else {
			// fallback: extract date from index filename (e.g., name_20251109.lrindex)
			baseName := filepath.Base(existingIndex)
			// find the date part (8 digits before .lrindex)
			if idx := strings.LastIndex(baseName, "_"); idx > 0 {
				datePart := strings.TrimSuffix(baseName[idx+1:], ".lrindex")
				if len(datePart) == 8 {
					indexedAt, err = time.Parse("20060102", datePart)
					if err != nil {
						return fmt.Errorf("cannot extract date from index filename: %w", err)
					}
				}
			}
			if indexedAt.IsZero() {
				// last resort: use file modification time
				info, err := os.Stat(existingIndex)
				if err != nil {
					return fmt.Errorf("cannot stat index file: %w", err)
				}
				indexedAt = info.ModTime()
			}
		}
		fmt.Printf("detecting changes since %s...\n", indexedAt.Format("2006-01-02 15:04:05"))
		changeSet, err = detectChangesMtime(srcPath, indexedAt, vs.Metadata.IndexedFiles, extensions)
		if err != nil {
			return fmt.Errorf("mtime change detection failed: %w", err)
		}
	}

	// report changes
	fmt.Printf("\n=== CHANGES DETECTED ===\n")
	fmt.Printf("Added:    %d files\n", len(changeSet.Added))
	fmt.Printf("Modified: %d files\n", len(changeSet.Modified))
	fmt.Printf("Deleted:  %d files\n", len(changeSet.Deleted))

	if !changeSet.HasChanges() {
		fmt.Println("\nno changes detected - index is up to date")
		return nil
	}

	// dry run - just show what would happen
	if dryRun {
		fmt.Println("\n=== DRY RUN ===")
		if len(changeSet.Added) > 0 {
			fmt.Println("Files to add:")
			for _, f := range changeSet.Added {
				fmt.Printf("  + %s\n", f)
			}
		}
		if len(changeSet.Modified) > 0 {
			fmt.Println("Files to re-index:")
			for _, f := range changeSet.Modified {
				fmt.Printf("  ~ %s\n", f)
			}
		}
		if len(changeSet.Deleted) > 0 {
			fmt.Println("Files to remove:")
			for _, f := range changeSet.Deleted {
				fmt.Printf("  - %s\n", f)
			}
		}
		return nil
	}

	// remove chunks from modified/deleted files
	toRemove := changeSet.RemovedFiles()
	if len(toRemove) > 0 {
		removed := vs.RemoveBySource(toRemove)
		fmt.Printf("removed %d chunks from %d changed/deleted files\n", removed, len(toRemove))
	}

	// load changed files
	changedFiles := changeSet.ChangedFiles()
	if len(changedFiles) > 0 {
		fmt.Printf("loading %d changed files...\n", len(changedFiles))
		loadResult, err := LoadSpecificFiles(srcPath, changedFiles, docType, maxFileSize, splitLarge)
		if err != nil {
			return fmt.Errorf("failed to load changed files: %w", err)
		}

		// chunk new documents
		var newChunks []Chunk
		for _, doc := range loadResult.Documents {
			docChunks := ChunkDocument(doc, maxChunkSize)
			newChunks = append(newChunks, docChunks...)
		}
		fmt.Printf("created %d new chunks\n", len(newChunks))

		if len(newChunks) > 0 {
			// generate embeddings for new chunks
			bar := progressbar.NewOptions(len(newChunks),
				progressbar.OptionSetDescription("generating embeddings"),
				progressbar.OptionShowCount(),
				progressbar.OptionSetWidth(40),
				progressbar.OptionThrottle(100*time.Millisecond),
				progressbar.OptionShowIts(),
				progressbar.OptionSetItsString("chunks"),
			)

			for _, chunk := range newChunks {
				embedding, err := llm.GetEmbedding(chunk.Text)
				if err != nil {
					return fmt.Errorf("failed to get embedding: %w", err)
				}
				vs.Add(chunk, embedding)
				bar.Add(1)
				time.Sleep(50 * time.Millisecond) // rate limit
			}
			bar.Finish()
			fmt.Println()
		}

		// update indexed files list
		// remove deleted files, add new files
		fileSet := make(map[string]bool)
		for _, f := range vs.Metadata.IndexedFiles {
			fileSet[f] = true
		}
		for _, f := range changeSet.Deleted {
			delete(fileSet, f)
		}
		for _, f := range changeSet.Added {
			fileSet[f] = true
		}
		vs.Metadata.IndexedFiles = make([]string, 0, len(fileSet))
		for f := range fileSet {
			vs.Metadata.IndexedFiles = append(vs.Metadata.IndexedFiles, f)
		}
	}

	// update metadata
	absPath, _ := filepath.Abs(srcPath)
	vs.Metadata.SourcePath = absPath
	vs.Metadata.IndexedAt = time.Now().Format(time.RFC3339)
	vs.Metadata.ChunkCount = len(vs.Chunks)
	vs.Metadata.FileCount = len(vs.Metadata.IndexedFiles)
	if useGit {
		commit, _ := getGitHeadCommit(srcPath)
		vs.Metadata.LastCommit = commit
	}

	// atomic save
	fmt.Printf("saving %s...\n", filepath.Base(finalOutPath))
	if err := atomicSave(vs, finalOutPath); err != nil {
		return fmt.Errorf("failed to save index: %w", err)
	}

	elapsed := time.Since(start)
	fmt.Printf("✓ incremental update complete (%d total chunks in %s)\n", len(vs.Chunks), elapsed.Round(time.Second))
	return nil
}

func printResults(question, answer string, results []SearchResult) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Printf("question: %s\n", question)
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("\nanswer:\n%s\n", answer)

	fmt.Println("\nsources:")
	for i, result := range results {
		fmt.Printf("  [%d] %s (similarity: %.3f)\n", i+1, result.Chunk.Source, result.Similarity)
	}
	fmt.Println()
}
