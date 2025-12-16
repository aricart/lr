package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// global preloaded resources
var (
	preloadedMSS *MultiSourceStore
	preloadedLLM LLMClient
	preloadMutex sync.RWMutex
)

func createMCPServer() *server.MCPServer {
	// create mcp server
	s := server.NewMCPServer(
		"localrag",
		"1.0.0",
		server.WithToolCapabilities(true),
	)

	// add query tool
	queryTool := mcp.NewTool("query_repositories",
		mcp.WithDescription("Query indexed code repositories and documentation. Returns relevant information from indexed sources."),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("The question to ask about the indexed repositories")),
		mcp.WithNumber("top_k",
			mcp.Description("Number of relevant chunks to retrieve (default: 3)")),
		mcp.WithBoolean("synthesize",
			mcp.Description("Use LLM to synthesize an answer from the chunks (default: true). Set to false to return raw chunks only.")),
		mcp.WithString("sources",
			mcp.Description("Comma-separated list of source names to search (e.g., 'jwt,nats-server'). If not specified, searches all sources.")),
	)

	s.AddTool(queryTool, handleQuery)

	// add list_indexes tool
	listTool := mcp.NewTool("list_indexes",
		mcp.WithDescription("List all available indexed repositories with metadata. Use this to see what's indexed before querying."),
	)
	s.AddTool(listTool, handleListIndexes)

	// add get_index_stats tool
	statsTool := mcp.NewTool("get_index_stats",
		mcp.WithDescription("Get detailed statistics about a specific index including file list."),
		mcp.WithString("name",
			mcp.Required(),
			mcp.Description("The index name (e.g., 'nats-server', 'docs')")),
	)
	s.AddTool(statsTool, handleGetIndexStats)

	// add search_by_file tool
	fileTool := mcp.NewTool("search_by_file",
		mcp.WithDescription("Get all indexed chunks from a specific file. Use this when user asks about a specific file rather than a concept."),
		mcp.WithString("path",
			mcp.Required(),
			mcp.Description("The file path to search for (can be partial, e.g., 'server.go' or 'cmd/main.go')")),
	)
	s.AddTool(fileTool, handleSearchByFile)

	// add get_diff_context tool for code review
	diffTool := mcp.NewTool("get_diff_context",
		mcp.WithDescription("Get git diff with relevant indexed context for code review. Requires an active review session (lr review start). Returns the uncommitted changes plus relevant code context from the review index."),
		mcp.WithNumber("top_k",
			mcp.Description("Number of relevant context chunks per changed file (default: 3)")),
	)
	s.AddTool(diffTool, handleGetDiffContext)

	return s
}

func handleQuery(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	// get arguments as map
	args, ok := request.Params.Arguments.(map[string]interface{})
	if !ok {
		return mcp.NewToolResultError("invalid arguments"), nil
	}

	// get query parameter
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return mcp.NewToolResultError("query parameter is required"), nil
	}

	// get top_k parameter (optional)
	topKVal := 3.0
	if topKArg, ok := args["top_k"]; ok {
		if topKFloat, ok := topKArg.(float64); ok {
			topKVal = topKFloat
		}
	}
	topK := int(topKVal)

	// get synthesize parameter (optional, default from env or true)
	synthesize := true
	if synthEnv := os.Getenv("LR_SYNTHESIZE"); synthEnv != "" {
		synthesize = synthEnv != "false"
	}
	if synthArg, ok := args["synthesize"]; ok {
		if synthBool, ok := synthArg.(bool); ok {
			synthesize = synthBool
		}
	}

	// get sources parameter (optional)
	var sources []string
	if sourcesArg, ok := args["sources"].(string); ok && sourcesArg != "" {
		for _, s := range strings.Split(sourcesArg, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				sources = append(sources, s)
			}
		}
	}

	// load vector store (always needed)
	var mss *MultiSourceStore
	var err error

	preloadMutex.RLock()
	if preloadedMSS != nil {
		mss = preloadedMSS
	}
	preloadMutex.RUnlock()

	if mss == nil {
		// load on-demand (no-preload mode)
		indexDir := getDefaultIndexDir()
		mss = NewMultiSourceStore(indexDir)
		if err := mss.LoadAll(); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("failed to load vector stores: %v", err)), nil
		}
	}

	if len(mss.Sources) == 0 {
		return mcp.NewToolResultError("no vector stores found. run 'lr index' to index repositories first"), nil
	}

	// if raw mode (no synthesis), we only need vector store
	if !synthesize {
		// get embedding for search (need minimal llm client just for embeddings)
		var llm LLMClient
		preloadMutex.RLock()
		if preloadedLLM != nil {
			llm = preloadedLLM
		}
		preloadMutex.RUnlock()

		if llm == nil {
			// temporarily redirect stdout to stderr to avoid polluting json-rpc
			oldStdout := os.Stdout
			os.Stdout = os.Stderr
			llm, err = getLLMClient()
			os.Stdout = oldStdout

			if err != nil {
				return mcp.NewToolResultError(fmt.Sprintf("failed to initialize LLM for embeddings: %v", err)), nil
			}
		}

		// get query embedding
		queryEmbedding, err := llm.GetEmbedding(query)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("failed to get query embedding: %v", err)), nil
		}

		// search for relevant chunks
		results := mss.Search(queryEmbedding, topK, sources)

		// format raw results
		var response string
		if len(sources) > 0 {
			response = fmt.Sprintf("searching %d of %d sources: %v\n\n", len(sources), len(mss.Sources), sources)
		} else {
			response = fmt.Sprintf("searching all %d sources: %v\n\n", len(mss.Sources), mss.ListSources())
		}
		response += fmt.Sprintf("================================================================================\n")
		response += fmt.Sprintf("query: %s\n", query)
		response += fmt.Sprintf("================================================================================\n\n")
		response += fmt.Sprintf("found %d relevant chunks:\n\n", len(results))

		for i, result := range results {
			response += fmt.Sprintf("--- chunk %d (source: %s, similarity: %.3f) ---\n", i+1, result.Chunk.Source, result.Similarity)
			response += result.Chunk.Text
			response += "\n\n"
		}

		return mcp.NewToolResultText(response), nil
	}

	// synthesized mode - need llm for chat
	var llm LLMClient
	preloadMutex.RLock()
	if preloadedLLM != nil {
		llm = preloadedLLM
	}
	preloadMutex.RUnlock()

	if llm == nil {
		// temporarily redirect stdout to stderr to avoid polluting json-rpc
		oldStdout := os.Stdout
		os.Stdout = os.Stderr
		llm, err = getLLMClient()
		os.Stdout = oldStdout

		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("failed to initialize LLM: %v", err)), nil
		}
	}

	// create rag and query
	rag := NewRAGMultiSource(mss, llm)
	answer, results, err := rag.QueryWithSources(query, topK, sources)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("query failed: %v", err)), nil
	}

	// format response
	var response string
	if len(sources) > 0 {
		response = fmt.Sprintf("searching %d of %d sources: %v\n\n", len(sources), len(mss.Sources), sources)
	} else {
		response = fmt.Sprintf("searching all %d sources: %v\n\n", len(mss.Sources), mss.ListSources())
	}
	response += fmt.Sprintf("================================================================================\n")
	response += fmt.Sprintf("question: %s\n", query)
	response += fmt.Sprintf("================================================================================\n\n")
	response += fmt.Sprintf("answer:\n%s\n\n", answer)
	response += fmt.Sprintf("sources:\n")
	for i, result := range results {
		response += fmt.Sprintf("  [%d] %s (similarity: %.3f)\n", i+1, result.Chunk.Source, result.Similarity)
	}

	return mcp.NewToolResultText(response), nil
}

func handleListIndexes(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	// use preloaded stores if available
	var mss *MultiSourceStore

	preloadMutex.RLock()
	if preloadedMSS != nil {
		mss = preloadedMSS
	}
	preloadMutex.RUnlock()

	if mss == nil {
		// load on-demand
		indexDir := getDefaultIndexDir()
		mss = NewMultiSourceStore(indexDir)
		if err := mss.LoadAll(); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("failed to load indexes: %v", err)), nil
		}
	}

	if len(mss.Sources) == 0 {
		return mcp.NewToolResultText("no indexes found. run 'lr index' to index repositories first."), nil
	}

	response := fmt.Sprintf("found %d indexed repositories:\n\n", len(mss.Sources))

	for name, vs := range mss.Sources {
		response += fmt.Sprintf("• %s\n", name)
		response += fmt.Sprintf("  chunks: %d\n", len(vs.Chunks))
		if vs.Metadata.FileCount > 0 {
			response += fmt.Sprintf("  files: %d\n", vs.Metadata.FileCount)
		}
		if vs.Metadata.SourcePath != "" {
			response += fmt.Sprintf("  source: %s\n", vs.Metadata.SourcePath)
		}
		if vs.Metadata.IndexedAt != "" {
			response += fmt.Sprintf("  indexed: %s\n", vs.Metadata.IndexedAt)
		}
		response += "\n"
	}

	return mcp.NewToolResultText(response), nil
}

func handleGetIndexStats(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	// get arguments
	args, ok := request.Params.Arguments.(map[string]interface{})
	if !ok {
		return mcp.NewToolResultError("invalid arguments"), nil
	}

	name, ok := args["name"].(string)
	if !ok || name == "" {
		return mcp.NewToolResultError("name parameter is required"), nil
	}

	// use preloaded stores if available
	var mss *MultiSourceStore

	preloadMutex.RLock()
	if preloadedMSS != nil {
		mss = preloadedMSS
	}
	preloadMutex.RUnlock()

	if mss == nil {
		// load on-demand
		indexDir := getDefaultIndexDir()
		mss = NewMultiSourceStore(indexDir)
		if err := mss.LoadAll(); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("failed to load indexes: %v", err)), nil
		}
	}

	// find the index (try exact match first, then partial)
	var vs *VectorStore
	var foundName string
	for n, store := range mss.Sources {
		if n == name {
			vs = store
			foundName = n
			break
		}
	}
	if vs == nil {
		// try partial match
		for n, store := range mss.Sources {
			if strings.Contains(strings.ToLower(n), strings.ToLower(name)) {
				vs = store
				foundName = n
				break
			}
		}
	}

	if vs == nil {
		available := make([]string, 0, len(mss.Sources))
		for n := range mss.Sources {
			available = append(available, n)
		}
		return mcp.NewToolResultError(fmt.Sprintf("index '%s' not found. available: %v", name, available)), nil
	}

	response := fmt.Sprintf("index: %s\n\n", foundName)
	response += fmt.Sprintf("chunks: %d\n", len(vs.Chunks))
	response += fmt.Sprintf("files: %d\n", vs.Metadata.FileCount)
	if vs.Metadata.SourcePath != "" {
		response += fmt.Sprintf("source path: %s\n", vs.Metadata.SourcePath)
	}
	if vs.Metadata.IndexedAt != "" {
		response += fmt.Sprintf("indexed at: %s\n", vs.Metadata.IndexedAt)
	}
	if vs.Metadata.LastCommit != "" {
		response += fmt.Sprintf("git commit: %s\n", vs.Metadata.LastCommit)
	}

	// list indexed files
	if len(vs.Metadata.IndexedFiles) > 0 {
		response += fmt.Sprintf("\nindexed files (%d):\n", len(vs.Metadata.IndexedFiles))
		for _, f := range vs.Metadata.IndexedFiles {
			response += fmt.Sprintf("  • %s\n", f)
		}
	}

	// list skipped files if any
	if len(vs.Metadata.SkippedFiles) > 0 {
		response += fmt.Sprintf("\nskipped files (%d):\n", len(vs.Metadata.SkippedFiles))
		for _, sf := range vs.Metadata.SkippedFiles {
			response += fmt.Sprintf("  • %s (%s)\n", sf.Path, sf.Reason)
		}
	}

	return mcp.NewToolResultText(response), nil
}

func handleSearchByFile(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	// get arguments
	args, ok := request.Params.Arguments.(map[string]interface{})
	if !ok {
		return mcp.NewToolResultError("invalid arguments"), nil
	}

	path, ok := args["path"].(string)
	if !ok || path == "" {
		return mcp.NewToolResultError("path parameter is required"), nil
	}

	// use preloaded stores if available
	var mss *MultiSourceStore

	preloadMutex.RLock()
	if preloadedMSS != nil {
		mss = preloadedMSS
	}
	preloadMutex.RUnlock()

	if mss == nil {
		// load on-demand
		indexDir := getDefaultIndexDir()
		mss = NewMultiSourceStore(indexDir)
		if err := mss.LoadAll(); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("failed to load indexes: %v", err)), nil
		}
	}

	// search all indexes for chunks matching the file path
	var matches []struct {
		source string
		chunk  Chunk
	}

	pathLower := strings.ToLower(path)
	for _, vs := range mss.Sources {
		for _, chunk := range vs.Chunks {
			if strings.Contains(strings.ToLower(chunk.Source), pathLower) {
				matches = append(matches, struct {
					source string
					chunk  Chunk
				}{source: chunk.Source, chunk: chunk})
			}
		}
	}

	if len(matches) == 0 {
		return mcp.NewToolResultText(fmt.Sprintf("no chunks found matching path '%s'", path)), nil
	}

	// group by source file
	byFile := make(map[string][]Chunk)
	for _, m := range matches {
		byFile[m.source] = append(byFile[m.source], m.chunk)
	}

	response := fmt.Sprintf("found %d chunks from %d files matching '%s':\n\n", len(matches), len(byFile), path)

	for file, chunks := range byFile {
		response += fmt.Sprintf("=== %s (%d chunks) ===\n\n", file, len(chunks))
		for i, chunk := range chunks {
			response += fmt.Sprintf("--- chunk %d ---\n", i+1)
			response += chunk.Text
			response += "\n\n"
		}
	}

	return mcp.NewToolResultText(response), nil
}

func handleGetDiffContext(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	// get arguments
	args, ok := request.Params.Arguments.(map[string]interface{})
	topK := 3
	if ok {
		if tk, ok := args["top_k"].(float64); ok {
			topK = int(tk)
		}
	}

	// load review session
	session, err := loadReviewSession()
	if err != nil {
		return mcp.NewToolResultError("no active review session. run 'lr review start' first"), nil
	}

	// get git diff from project directory (--no-ext-diff ensures unified format regardless of user config)
	cmd := exec.CommandContext(ctx, "git", "-C", session.ProjectPath, "diff", "--no-ext-diff")
	diffOutput, err := cmd.Output()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("failed to get git diff: %v", err)), nil
	}

	// also get staged changes
	cmdStaged := exec.CommandContext(ctx, "git", "-C", session.ProjectPath, "diff", "--cached", "--no-ext-diff")
	stagedOutput, _ := cmdStaged.Output()

	// combine diff output
	fullDiff := string(diffOutput)
	if len(stagedOutput) > 0 {
		fullDiff += "\n=== STAGED CHANGES ===\n" + string(stagedOutput)
	}

	if fullDiff == "" {
		return mcp.NewToolResultText("no uncommitted changes found"), nil
	}

	// extract changed file paths from diff
	changedFiles := extractChangedFiles(fullDiff)
	if len(changedFiles) == 0 {
		return mcp.NewToolResultText("git diff:\n\n" + fullDiff), nil
	}

	// load review index
	store := NewVectorStore()
	if err := store.Load(session.IndexPath); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("failed to load review index: %v", err)), nil
	}

	// build response with diff and context
	response := "=== GIT DIFF ===\n\n" + fullDiff + "\n\n"
	response += "=== RELEVANT CONTEXT ===\n\n"

	// for each changed file, find related context
	for _, file := range changedFiles {
		// search for this file in the index
		fileChunks := []Chunk{}
		for _, chunk := range store.Chunks {
			if strings.Contains(chunk.Source, file) {
				fileChunks = append(fileChunks, chunk)
			}
		}

		if len(fileChunks) > 0 {
			response += fmt.Sprintf("--- context from %s ---\n", file)
			for i, chunk := range fileChunks {
				if i >= topK {
					break
				}
				response += chunk.Text + "\n\n"
			}
		}
	}

	return mcp.NewToolResultText(response), nil
}

// extractChangedFiles parses a git diff and returns the list of changed file paths
func extractChangedFiles(diff string) []string {
	files := make(map[string]bool)
	lines := strings.Split(diff, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "+++ b/") {
			file := strings.TrimPrefix(line, "+++ b/")
			files[file] = true
		} else if strings.HasPrefix(line, "--- a/") {
			file := strings.TrimPrefix(line, "--- a/")
			if file != "/dev/null" {
				files[file] = true
			}
		}
	}

	result := make([]string, 0, len(files))
	for f := range files {
		result = append(result, f)
	}
	sort.Strings(result)
	return result
}

func reloadVectorStores() error {
	indexDir := getDefaultIndexDir()
	mss := NewMultiSourceStore(indexDir)
	if err := mss.LoadAll(); err != nil {
		return fmt.Errorf("failed to reload vector stores: %w", err)
	}

	preloadMutex.Lock()
	preloadedMSS = mss
	preloadMutex.Unlock()

	log.SetOutput(os.Stderr)
	log.Printf("reloaded %d vector store sources: %v", len(mss.Sources), mss.ListSources())
	log.SetOutput(nil)

	return nil
}

// reloadAllProcesses finds all lr processes and sends SIGUSR1 to them
func reloadAllProcesses() error {
	myPid := os.Getpid()

	// use pgrep to find lr processes
	cmd := exec.Command("pgrep", "-f", "lr mcp")
	output, err := cmd.Output()
	if err != nil {
		// pgrep returns exit code 1 if no processes found
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
			fmt.Println("no lr mcp processes found")
			return nil
		}
		return fmt.Errorf("failed to find lr processes: %w", err)
	}

	var signaled int
	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		pidStr := strings.TrimSpace(scanner.Text())
		if pidStr == "" {
			continue
		}

		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			continue
		}

		// skip our own process
		if pid == myPid {
			continue
		}

		process, err := os.FindProcess(pid)
		if err != nil {
			fmt.Printf("warning: could not find process %d: %v\n", pid, err)
			continue
		}

		if err := process.Signal(syscall.SIGUSR1); err != nil {
			fmt.Printf("warning: could not signal process %d: %v\n", pid, err)
			continue
		}

		fmt.Printf("sent reload signal to pid %d\n", pid)
		signaled++
	}

	if signaled == 0 {
		fmt.Println("no lr mcp processes found to reload")
	} else {
		fmt.Printf("reloaded %d process(es)\n", signaled)
	}

	return nil
}

func serveMCP() error {
	// handle --reload flag
	if reloadPid > 0 {
		// send SIGUSR1 to the specified pid
		process, err := os.FindProcess(reloadPid)
		if err != nil {
			return fmt.Errorf("failed to find process %d: %w", reloadPid, err)
		}

		if err := process.Signal(syscall.SIGUSR1); err != nil {
			return fmt.Errorf("failed to send reload signal to pid %d: %w", reloadPid, err)
		}

		fmt.Printf("sent reload signal to pid %d\n", reloadPid)
		return nil
	}

	// handle --reload-all flag
	if reloadAll {
		return reloadAllProcesses()
	}

	// suppress info logs to stderr (MCP uses stdout for protocol)
	log.SetOutput(nil)

	// preload resources unless --no-preload flag is set
	if !noPreload {
		// preload llm client
		llm, err := getLLMClient()
		if err != nil {
			return fmt.Errorf("failed to preload LLM client: %w", err)
		}
		preloadMutex.Lock()
		preloadedLLM = llm
		preloadMutex.Unlock()

		// preload vector stores
		if err := reloadVectorStores(); err != nil {
			return err
		}
	}

	// setup signal handler for reload
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGUSR1)

	go func() {
		for range sigChan {
			log.SetOutput(os.Stderr)
			log.Println("received reload signal, reloading vector stores...")
			log.SetOutput(nil)

			if err := reloadVectorStores(); err != nil {
				log.SetOutput(os.Stderr)
				log.Printf("error reloading: %v", err)
				log.SetOutput(nil)
			}
		}
	}()

	// print pid so user knows how to reload
	log.SetOutput(os.Stderr)
	log.Printf("mcp server started (pid: %d)", os.Getpid())
	log.Printf("to reload indexes: lr mcp --reload %d", os.Getpid())
	log.SetOutput(nil)

	mcpServer := createMCPServer()

	if err := server.ServeStdio(mcpServer); err != nil {
		return fmt.Errorf("mcp server error: %w", err)
	}

	return nil
}
