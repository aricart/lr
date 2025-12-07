package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
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
		mcp.WithDescription("Query indexed code repositories and documentation. Returns relevant information from all indexed sources."),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("The question to ask about the indexed repositories")),
		mcp.WithNumber("top_k",
			mcp.Description("Number of relevant chunks to retrieve (default: 3)")),
		mcp.WithBoolean("synthesize",
			mcp.Description("Use LLM to synthesize an answer from the chunks (default: true). Set to false to return raw chunks only.")),
	)

	s.AddTool(queryTool, handleQuery)

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
		results := mss.Search(queryEmbedding, topK, []string{})

		// format raw results
		response := fmt.Sprintf("loaded %d sources: %v\n\n", len(mss.Sources), mss.ListSources())
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
	answer, results, err := rag.Query(query, topK)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("query failed: %v", err)), nil
	}

	// format response
	response := fmt.Sprintf("loaded %d sources: %v\n\n", len(mss.Sources), mss.ListSources())
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
