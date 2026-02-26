package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIndexing(t *testing.T) {
	// create a temporary test directory
	tmpDir := t.TempDir()
	testSrcDir := filepath.Join(tmpDir, "src")
	if err := os.MkdirAll(testSrcDir, 0755); err != nil {
		t.Fatalf("failed to create test dir: %v", err)
	}

	// copy one file from the project to test with
	testFile := filepath.Join(testSrcDir, "test.go")
	testContent := `package main

import "fmt"

// TestFunc is a test function
func TestFunc(x int) int {
	return x * 2
}

func main() {
	fmt.Println(TestFunc(5))
}
`
	if err := os.WriteFile(testFile, []byte(testContent), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	// test output path
	outputFile := filepath.Join(tmpDir, "test.lrindex")

	// create a mock LLM client that returns dummy embeddings
	mockLLM := &MockLLMClient{}

	// load the test files
	loader := func(dir string) ([]Document, error) {
		return LoadCodeFiles(dir)
	}

	// run indexing
	t.Logf("output file will be: %s", outputFile)
	checkpointFile := filepath.Join(tmpDir, "test.checkpoint.lrindex")
	t.Logf("checkpoint file should be: %s", checkpointFile)

	err := indexSingleSource(mockLLM, testSrcDir, outputFile, loader)
	if err != nil {
		t.Fatalf("indexing failed: %v", err)
	}

	// check what files exist in the output directory
	entries, _ := os.ReadDir(tmpDir)
	t.Logf("files in tmpDir after indexing:")
	for _, e := range entries {
		info, _ := e.Info()
		t.Logf("  - %s (size: %d)", e.Name(), info.Size())
	}

	// verify the output file was created
	if _, err := os.Stat(outputFile); os.IsNotExist(err) {
		t.Fatalf("output file was not created: %s", outputFile)
	}

	// verify we can load the index back
	vs := NewVectorStore()
	if err := vs.Load(outputFile); err != nil {
		t.Fatalf("failed to load index: %v", err)
	}

	if len(vs.Chunks) == 0 {
		t.Fatal("index has no chunks")
	}

	t.Logf("successfully indexed %d chunks", len(vs.Chunks))
}

func TestIndexingRust(t *testing.T) {
	tmpDir := t.TempDir()
	testSrcDir := filepath.Join(tmpDir, "src")
	if err := os.MkdirAll(testSrcDir, 0755); err != nil {
		t.Fatalf("failed to create test dir: %v", err)
	}

	testFile := filepath.Join(testSrcDir, "main.rs")
	testContent := `use std::collections::HashMap;

pub struct Config {
    name: String,
    values: HashMap<String, String>,
}

impl Config {
    pub fn new(name: &str) -> Self {
        Config {
            name: name.to_string(),
            values: HashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&String> {
        self.values.get(key)
    }
}

fn main() {
    let config = Config::new("test");
    println!("{:?}", config.get("key"));
}
`
	if err := os.WriteFile(testFile, []byte(testContent), 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	outputFile := filepath.Join(tmpDir, "test.lrindex")
	mockLLM := &MockLLMClient{}

	loader := func(dir string) ([]Document, error) {
		return LoadCodeFiles(dir)
	}

	err := indexSingleSource(mockLLM, testSrcDir, outputFile, loader)
	if err != nil {
		t.Fatalf("indexing failed: %v", err)
	}

	vs := NewVectorStore()
	if err := vs.Load(outputFile); err != nil {
		t.Fatalf("failed to load index: %v", err)
	}

	if len(vs.Chunks) == 0 {
		t.Fatal("index has no chunks")
	}

	// verify chunks have rust type
	for _, chunk := range vs.Chunks {
		if chunk.Metadata["type"] != "rust" {
			t.Errorf("expected chunk type 'rust', got '%s'", chunk.Metadata["type"])
		}
	}

	t.Logf("successfully indexed %d rust chunks", len(vs.Chunks))
}

func TestRustFileTypeDetection(t *testing.T) {
	tmpDir := t.TempDir()

	rsFile := filepath.Join(tmpDir, "lib.rs")
	rsContent := `pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
`
	if err := os.WriteFile(rsFile, []byte(rsContent), 0644); err != nil {
		t.Fatalf("failed to write .rs file: %v", err)
	}

	result, err := LoadFilesByExtensionsWithStats(tmpDir, []string{".rs"}, "code", 100*1024)
	if err != nil {
		t.Fatalf("failed to load files: %v", err)
	}

	if len(result.Documents) != 1 {
		t.Fatalf("expected 1 document, got %d", len(result.Documents))
	}

	doc := result.Documents[0]
	if doc.Metadata["type"] != "rust" {
		t.Errorf("expected type 'rust', got '%s'", doc.Metadata["type"])
	}
}

func TestRustChunking(t *testing.T) {
	content := `use std::io;

pub struct Server {
    port: u16,
    host: String,
}

impl Server {
    pub fn new(host: &str, port: u16) -> Self {
        Server {
            host: host.to_string(),
            port,
        }
    }

    pub fn start(&self) -> io::Result<()> {
        println!("Starting server on {}:{}", self.host, self.port);
        Ok(())
    }
}

fn main() {
    let server = Server::new("localhost", 8080);
    server.start().unwrap();
}
`
	doc := Document{
		Content: content,
		Source:  "server.rs",
		Metadata: map[string]string{
			"path": "server.rs",
			"type": "rust",
		},
	}

	chunks := ChunkDocument(doc, 1500)
	if len(chunks) == 0 {
		t.Fatal("expected chunks from rust file, got none")
	}

	// verify function-based splitting produced multiple chunks
	if len(chunks) < 2 {
		t.Errorf("expected multiple chunks from function splitting, got %d", len(chunks))
	}

	for _, chunk := range chunks {
		if chunk.Metadata["type"] != "rust" {
			t.Errorf("expected chunk type 'rust', got '%s'", chunk.Metadata["type"])
		}
	}

	t.Logf("rust chunking produced %d chunks", len(chunks))
}

// MockLLMClient implements LLMClient for testing
type MockLLMClient struct{}

func (m *MockLLMClient) GetEmbedding(text string) ([]float64, error) {
	// return a dummy embedding vector (1536 dimensions like OpenAI)
	embedding := make([]float64, 1536)
	for i := range embedding {
		embedding[i] = 0.1
	}
	return embedding, nil
}

func (m *MockLLMClient) Chat(messages []Message) (string, error) {
	return "mock response", nil
}
