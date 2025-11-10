package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestVectorStoreSave(t *testing.T) {
	// create a simple vector store
	vs := NewVectorStore()
	vs.Add(Chunk{
		Text:   "test chunk",
		Source: "test.go",
	}, []float64{0.1, 0.2, 0.3})

	// save to temp file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.lrindex")

	t.Logf("saving to: %s", testFile)
	err := vs.Save(testFile)
	if err != nil {
		t.Fatalf("save failed: %v", err)
	}

	// check if file exists
	info, err := os.Stat(testFile)
	if os.IsNotExist(err) {
		// list what's in the directory
		entries, _ := os.ReadDir(tmpDir)
		t.Logf("files in tmpDir:")
		for _, e := range entries {
			t.Logf("  - %s", e.Name())
		}
		t.Fatalf("file was not created: %s", testFile)
	}
	if err != nil {
		t.Fatalf("stat failed: %v", err)
	}

	t.Logf("file created successfully, size: %d bytes", info.Size())

	// try to load it back
	vs2 := NewVectorStore()
	if err := vs2.Load(testFile); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	if len(vs2.Chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(vs2.Chunks))
	}

	t.Log("save/load test passed!")
}