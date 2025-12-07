package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
)

const (
	vectorStoreFile    = "nats_vectorstore.json"
	checkpointFile     = "nats_vectorstore.checkpoint.json"
	docsDir            = "nats.docs"
	natsGoDir          = "nats.go"
	natsJsDir          = "nats.js"
	natsServerDir      = "nats-server"
	nscDir             = "nsc"
	natscliDir         = "natscli"
	jwtAuthBuilderDir  = "jwt-auth-builder.go"
	nstGoDir           = "nst.go"
	jsperftoolDir      = "jsperftool"
	maxChunkSize       = 1500
	maxChunkTokens     = 6000 // openai limit is 8192, leaving buffer
	checkpointInterval = 100  // save every 100 chunks
)

// source represents a documentation/code source to index
type source struct {
	dir    string
	id     string // unique identifier for vector store file
	loader func(string) ([]Document, error)
	name   string
}

var sources = []source{
	{dir: docsDir, id: "docs", loader: LoadMarkdownFiles, name: "NATS Documentation"},
	{dir: natsGoDir, id: "nats-go", loader: LoadCodeFiles, name: "NATS Go Client"},
	{dir: natsJsDir, id: "nats-js", loader: LoadCodeFiles, name: "NATS JavaScript Client"},
	{dir: nscDir, id: "nsc", loader: LoadCodeFiles, name: "NSC (NATS Configuration Tool)"},
	{dir: natscliDir, id: "natscli", loader: LoadCodeFiles, name: "NATS CLI"},
	{dir: jwtAuthBuilderDir, id: "jwt-auth-builder", loader: LoadCodeFiles, name: "JWT Auth Builder"},
	{dir: nstGoDir, id: "nst", loader: LoadCodeFiles, name: "NST (NATS Server Test Utilities)"},
	{dir: jsperftoolDir, id: "jsperftool", loader: LoadCodeFiles, name: "JetStream Perf Tool"},
}

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

	// query command flags
	queryText    string
	topK         int
	querySources []string
	useMCP       bool
	noSynthesize bool

	// mcp command flags
	noPreload bool
	reloadPid int
	reloadAll bool
)

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
	indexCmd.Flags().BoolVar(&useCode, "code", false, "index code files (.go, .js, .ts, etc)")
	indexCmd.Flags().BoolVar(&useDocs, "docs", false, "index documentation files (.md)")
	indexCmd.Flags().StringVar(&outPath, "out", "", "exact output path (e.g., indexes/myindex.lrindex)")
	indexCmd.Flags().StringVar(&outName, "out-name", "", "output name (saved as indexes/{name}_YYYYMMDD.lrindex)")
	indexCmd.Flags().BoolVar(&dryRun, "dry-run", false, "show what would be indexed without actually indexing")
	indexCmd.Flags().Int64Var(&maxFileSize, "max-file-size", 100*1024, "maximum file size in bytes (default 100KB)")
	indexCmd.Flags().BoolVar(&splitLarge, "split-large", false, "split large files into sections instead of skipping them")
	indexCmd.Flags().BoolVar(&includeTests, "include-tests", false, "include test files (often contain useful usage examples)")
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

	// add commands
	rootCmd.AddCommand(indexCmd)
	rootCmd.AddCommand(queryCmd)
	rootCmd.AddCommand(interactiveCmd)
	rootCmd.AddCommand(mcpCmd)
	rootCmd.AddCommand(listCmd)
	rootCmd.AddCommand(setupCmd)
	rootCmd.AddCommand(pathsCmd)
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

	// priority order for embedding+chat combinations
	if voyageKey != "" && claudeKey != "" {
		fmt.Println("using voyage ai embeddings + claude chat (recommended for code!)")
		return NewVoyageClaudeClient(voyageKey, claudeKey), nil
	} else if openaiKey != "" && claudeKey != "" {
		fmt.Println("using openai embeddings + claude chat")
		return NewHybridClient(openaiKey, claudeKey), nil
	} else if openaiKey != "" {
		fmt.Println("using openai for embeddings and chat")
		return NewOpenAIClient(openaiKey), nil
	}

	return nil, fmt.Errorf("no api key found. please set one of:\n" +
		"  - OPENAI_API_KEY (for openai only)\n" +
		"  - OPENAI_API_KEY + ANTHROPIC_API_KEY (hybrid mode)\n" +
		"  - VOYAGE_API_KEY + ANTHROPIC_API_KEY (recommended for code!)")
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

func runIndex(cmd *cobra.Command, args []string) error {
	// validate flags
	if !dryRun {
		if outPath == "" && outName == "" {
			return fmt.Errorf("either --out or --out-name is required when not using --dry-run")
		}
		if outPath != "" && outName != "" {
			return fmt.Errorf("cannot specify both --out and --out-name")
		}
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

func runQuery(cmd *cobra.Command, args []string) error {
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

func runList(cmd *cobra.Command, args []string) error {
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

func runInteractive(cmd *cobra.Command, args []string) error {
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

func runMCP(cmd *cobra.Command, args []string) error {
	return serveMCP()
}

func runPaths(cmd *cobra.Command, args []string) {
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

func runSetup(cmd *cobra.Command, args []string) error {
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

func indexDocumentation(llm LLMClient) error {
	start := time.Now()

	totalChunks := 0

	// index each source separately
	for _, src := range sources {
		fmt.Printf("\n=== indexing %s ===\n", src.name)

		// check if directory exists
		if _, err := os.Stat(src.dir); os.IsNotExist(err) {
			fmt.Printf("warning: %s directory not found, skipping\n", src.dir)
			continue
		}

		// load files
		fmt.Printf("loading files from %s...\n", src.dir)
		docs, err := src.loader(src.dir)
		if err != nil {
			return fmt.Errorf("failed to load %s: %w", src.name, err)
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
		totalChunks += len(chunks)

		// create embeddings for this source
		if err := indexSource(llm, src.id, chunks); err != nil {
			return fmt.Errorf("failed to index %s: %w", src.name, err)
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("\nindexing complete! indexed %d chunks in %s\n", totalChunks, elapsed.Round(time.Second))

	return nil
}

func indexSource(llm LLMClient, sourceID string, chunks []Chunk) error {
	vectorStoreDir := "vectorstore"
	// ensure directory exists
	if err := os.MkdirAll(vectorStoreDir, 0755); err != nil {
		return fmt.Errorf("failed to create vectorstore directory: %w", err)
	}

	timestamp := time.Now().Format("20060102")
	checkpointFile := filepath.Join(vectorStoreDir, fmt.Sprintf("nats_%s.checkpoint.json", sourceID))
	outputFile := filepath.Join(vectorStoreDir, fmt.Sprintf("nats_%s_%s.json", sourceID, timestamp))

	// try to load checkpoint if it exists
	vs := NewVectorStore()
	startIdx := 0

	if _, err := os.Stat(checkpointFile); err == nil {
		fmt.Printf("found checkpoint for %s, resuming...\n", sourceID)
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
			progressbar.OptionSetDescription(fmt.Sprintf("generating embeddings (%s)", sourceID)),
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
			return fmt.Errorf("failed to get embedding for chunk %d (source: %s, size: %d chars, ~%d tokens): %w",
				i, chunk.Source, len(chunk.Text), len(chunk.Text)/4, err)
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

	// save final vector store
	fmt.Printf("saving %s...\n", outputFile)
	if err := vs.Save(outputFile); err != nil {
		return fmt.Errorf("failed to save vector store: %w", err)
	}

	// remove checkpoint file since we completed successfully
	if _, err := os.Stat(checkpointFile); err == nil {
		os.Remove(checkpointFile)
	}

	fmt.Printf("✓ %s indexed successfully (%d chunks)\n", sourceID, len(chunks))
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
