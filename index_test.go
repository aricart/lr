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
