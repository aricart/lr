package main

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// MultiSourceStore manages multiple independent vector stores
type MultiSourceStore struct {
	Sources map[string]*VectorStore
	BaseDir string
}

// NewMultiSourceStore creates a new multi-source store
func NewMultiSourceStore(baseDir string) *MultiSourceStore {
	return &MultiSourceStore{
		Sources: make(map[string]*VectorStore),
		BaseDir: baseDir,
	}
}

// LoadSource loads a specific source's vector store (most recent version)
func (m *MultiSourceStore) LoadSource(name string) error {
	// try multiple filename patterns to find the source (.lrindex preferred, .json for backward compat)
	patterns := []string{
		filepath.Join(m.BaseDir, fmt.Sprintf("%s*.lrindex", name)),
		filepath.Join(m.BaseDir, fmt.Sprintf("*_%s*.lrindex", name)),
		filepath.Join(m.BaseDir, fmt.Sprintf("%s*.json", name)),
		filepath.Join(m.BaseDir, fmt.Sprintf("*_%s*.json", name)),
	}

	var allFiles []string
	for _, pattern := range patterns {
		files, err := filepath.Glob(pattern)
		if err != nil {
			return err
		}
		allFiles = append(allFiles, files...)
	}

	// filter out checkpoint files
	var validFiles []string
	for _, file := range allFiles {
		if !strings.Contains(file, "checkpoint") {
			validFiles = append(validFiles, file)
		}
	}

	if len(validFiles) == 0 {
		return fmt.Errorf("no vector store found for source %s", name)
	}

	// sort by filename (newest timestamp last)
	sort.Strings(validFiles)
	mostRecent := validFiles[len(validFiles)-1]

	vs := NewVectorStore()
	if err := vs.Load(mostRecent); err != nil {
		return fmt.Errorf("failed to load source %s: %w", name, err)
	}

	m.Sources[name] = vs
	return nil
}

// SaveSource saves a specific source's vector store
func (m *MultiSourceStore) SaveSource(name string, vs *VectorStore) error {
	filepath := filepath.Join(m.BaseDir, fmt.Sprintf("%s.lrindex", name))

	if err := vs.Save(filepath); err != nil {
		return fmt.Errorf("failed to save source %s: %w", name, err)
	}

	m.Sources[name] = vs
	return nil
}

// LoadAll loads all available source vector stores
func (m *MultiSourceStore) LoadAll() error {
	// list all index files (.lrindex and .json for backward compat)
	patterns := []string{
		filepath.Join(m.BaseDir, "*.lrindex"),
		filepath.Join(m.BaseDir, "*.json"),
	}
	var files []string
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return err
		}
		files = append(files, matches...)
	}

	// group files by source name
	sourceNames := make(map[string]bool)
	for _, file := range files {
		base := filepath.Base(file)

		// skip checkpoint files
		if strings.Contains(base, "checkpoint") {
			continue
		}

		// extract source name (strip extension and timestamp if present)
		name := strings.TrimSuffix(base, ".lrindex")
		name = strings.TrimSuffix(name, ".json")

		// strip common prefixes from filename
		for _, prefix := range []string{"nats_", "lr_"} {
			if strings.HasPrefix(name, prefix) {
				name = strings.TrimPrefix(name, prefix)
				break
			}
		}

		// if it has a timestamp suffix, remove it
		if parts := strings.Split(name, "_"); len(parts) > 1 {
			// check if last part looks like a date (8 digits)
			lastPart := parts[len(parts)-1]
			if len(lastPart) == 8 {
				name = strings.Join(parts[:len(parts)-1], "_")
			}
		}

		sourceNames[name] = true
	}

	// load each unique source
	for name := range sourceNames {
		if err := m.LoadSource(name); err != nil {
			return err
		}
	}

	return nil
}

// Search searches across specified sources (or all if empty)
func (m *MultiSourceStore) Search(queryEmbedding []float64, topK int, sources []string) []SearchResult {
	var allResults []SearchResult

	// if no sources specified, search all
	if len(sources) == 0 {
		for name := range m.Sources {
			sources = append(sources, name)
		}
	}

	// search each specified source
	for _, sourceName := range sources {
		vs, ok := m.Sources[sourceName]
		if !ok {
			continue
		}

		results := vs.Search(queryEmbedding, topK)

		// add source name to metadata
		for i := range results {
			if results[i].Chunk.Metadata == nil {
				results[i].Chunk.Metadata = make(map[string]string)
			}
			results[i].Chunk.Metadata["vector_source"] = sourceName
		}

		allResults = append(allResults, results...)
	}

	// sort by similarity and take top k
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Similarity > allResults[j].Similarity
	})

	if topK > len(allResults) {
		topK = len(allResults)
	}

	return allResults[:topK]
}

// ListSources returns all available source names
func (m *MultiSourceStore) ListSources() []string {
	var names []string
	for name := range m.Sources {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// GetSourceStats returns statistics about each source
func (m *MultiSourceStore) GetSourceStats() map[string]int {
	stats := make(map[string]int)
	for name, vs := range m.Sources {
		stats[name] = len(vs.Chunks)
	}
	return stats
}

// SourceExists checks if a source vector store file exists
func SourceExists(baseDir, name string) bool {
	// check multiple possible filename patterns (.lrindex and .json)
	patterns := []string{
		filepath.Join(baseDir, fmt.Sprintf("%s.lrindex", name)),
		filepath.Join(baseDir, fmt.Sprintf("%s_*.lrindex", name)),
		filepath.Join(baseDir, fmt.Sprintf("*_%s.lrindex", name)),
		filepath.Join(baseDir, fmt.Sprintf("%s.json", name)),
		filepath.Join(baseDir, fmt.Sprintf("%s_*.json", name)),
		filepath.Join(baseDir, fmt.Sprintf("*_%s.json", name)),
	}

	for _, pattern := range patterns {
		matches, _ := filepath.Glob(pattern)
		if len(matches) > 0 {
			return true
		}
	}
	return false
}
