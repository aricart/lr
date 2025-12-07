package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	ignore "github.com/sabhiram/go-gitignore"
)

// Document represents a loaded document with metadata
type Document struct {
	Content  string
	Source   string
	Metadata map[string]string
}

// LoadResult contains documents and metadata about the loading process
type LoadResult struct {
	Documents    []Document
	SkippedFiles []SkippedFile
	TotalFiles   int
}

// LoadMarkdownFiles loads all markdown files from the given directory
func LoadMarkdownFiles(rootDir string) ([]Document, error) {
	return LoadFilesByExtensions(rootDir, []string{".md"}, "markdown")
}

// LoadCodeFiles loads code files (Go, JavaScript, TypeScript, Python, Java, C) from the given directory
func LoadCodeFiles(rootDir string) ([]Document, error) {
	return LoadFilesByExtensions(rootDir, []string{".go", ".js", ".ts", ".jsx", ".tsx", ".py", ".java", ".c", ".h"}, "code")
}

// LoadFilesByExtensions loads files with specific extensions from the given directory
func LoadFilesByExtensions(rootDir string, extensions []string, docType string) ([]Document, error) {
	result, err := LoadFilesByExtensionsWithStats(rootDir, extensions, docType, 100*1024)
	return result.Documents, err
}

// LoadFilesByExtensionsWithStats loads files and returns detailed statistics
func LoadFilesByExtensionsWithStats(rootDir string, extensions []string, docType string, maxFileSize int64) (LoadResult, error) {
	return LoadFilesByExtensionsWithStatsAndSplit(rootDir, extensions, docType, maxFileSize, false, false)
}

// LoadFilesByExtensionsWithStatsAndSplit loads files with option to split large files
func LoadFilesByExtensionsWithStatsAndSplit(rootDir string, extensions []string, docType string, maxFileSize int64, splitLarge bool, includeTests bool) (LoadResult, error) {
	result := LoadResult{
		Documents:    []Document{},
		SkippedFiles: []SkippedFile{},
	}

	// try to load .gitignore if it exists
	var gitignore *ignore.GitIgnore
	gitignorePath := filepath.Join(rootDir, ".gitignore")
	if _, err := os.Stat(gitignorePath); err == nil {
		gitignore, _ = ignore.CompileIgnoreFile(gitignorePath)
	}

	err := filepath.WalkDir(rootDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// get relative path for gitignore checking
		relPath, _ := filepath.Rel(rootDir, path)

		// check gitignore for files only - don't skip directories based on gitignore
		// because allowlist patterns (like "* then !*.go") need to check actual files
		if gitignore != nil && !d.IsDir() && gitignore.MatchesPath(relPath) {
			return nil
		}

		// skip directories
		if d.IsDir() {
			// skip common directories we don't want to index
			dirName := d.Name()
			if dirName == "node_modules" || dirName == ".git" || dirName == "vendor" ||
				dirName == "dist" || dirName == "build" || dirName == ".github" ||
				dirName == "docs" || dirName == "gitbook" || dirName == "assets" {
				return filepath.SkipDir
			}
			return nil
		}

		result.TotalFiles++

		// check if file has one of the desired extensions
		hasExtension := false
		for _, ext := range extensions {
			if strings.HasSuffix(strings.ToLower(path), ext) {
				hasExtension = true
				break
			}
		}

		if !hasExtension {
			// track as skipped with extension reason
			ext := filepath.Ext(path)
			if ext != "" {
				result.SkippedFiles = append(result.SkippedFiles, SkippedFile{
					Path:   relPath,
					Reason: fmt.Sprintf("wrong extension (%s)", ext),
					Size:   0,
				})
			}
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return err
		}

		// skip test files unless includeTests is true
		baseName := filepath.Base(path)
		if !includeTests && (strings.HasSuffix(baseName, "_test.go") ||
			strings.HasSuffix(baseName, "_test.ts") || strings.HasSuffix(baseName, "_test.js") ||
			strings.HasSuffix(baseName, ".test.ts") || strings.HasSuffix(baseName, ".test.js") ||
			strings.HasSuffix(baseName, "_test.py") || strings.HasSuffix(baseName, "Test.java") ||
			strings.Contains(baseName, "test_")) {
			result.SkippedFiles = append(result.SkippedFiles, SkippedFile{
				Path:   relPath,
				Reason: "test file",
				Size:   info.Size(),
			})
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		// determine file type
		fileType := docType
		if strings.HasSuffix(path, ".go") {
			fileType = "go"
		} else if strings.HasSuffix(path, ".js") || strings.HasSuffix(path, ".jsx") {
			fileType = "javascript"
		} else if strings.HasSuffix(path, ".ts") || strings.HasSuffix(path, ".tsx") {
			fileType = "typescript"
		} else if strings.HasSuffix(path, ".templ") {
			fileType = "templ"
		} else if strings.HasSuffix(path, ".py") {
			fileType = "python"
		} else if strings.HasSuffix(path, ".java") {
			fileType = "java"
		} else if strings.HasSuffix(path, ".c") || strings.HasSuffix(path, ".h") {
			fileType = "c"
		}

		// handle large files
		if int64(len(content)) > maxFileSize {
			if splitLarge {
				// split large file into multiple documents
				splitDocs := splitLargeFile(string(content), relPath, fileType, int(maxFileSize))
				result.Documents = append(result.Documents, splitDocs...)
				fmt.Printf("  split large file: %s into %d parts\n", relPath, len(splitDocs))
				return nil
			} else {
				result.SkippedFiles = append(result.SkippedFiles, SkippedFile{
					Path:   relPath,
					Reason: fmt.Sprintf("too large (%dKB, max %dKB)", len(content)/1024, maxFileSize/1024),
					Size:   int64(len(content)),
				})
				return nil
			}
		}

		doc := Document{
			Content: string(content),
			Source:  relPath,
			Metadata: map[string]string{
				"path": relPath,
				"type": fileType,
			},
		}

		result.Documents = append(result.Documents, doc)
		return nil
	})

	return result, err
}

// splitLargeFile splits a large file into multiple documents
func splitLargeFile(content, path, fileType string, maxSize int) []Document {
	var docs []Document

	// simple strategy: split by lines, keeping chunks under maxSize
	lines := strings.Split(content, "\n")
	var currentChunk strings.Builder
	partNum := 1

	for _, line := range lines {
		// if adding this line would exceed max size, save current chunk
		if currentChunk.Len()+len(line)+1 > maxSize && currentChunk.Len() > 0 {
			docs = append(docs, Document{
				Content: currentChunk.String(),
				Source:  fmt.Sprintf("%s (part %d)", path, partNum),
				Metadata: map[string]string{
					"path": path,
					"type": fileType,
					"part": fmt.Sprintf("%d", partNum),
				},
			})
			currentChunk.Reset()
			partNum++
		}

		currentChunk.WriteString(line)
		currentChunk.WriteString("\n")
	}

	// add final chunk if non-empty
	if currentChunk.Len() > 0 {
		docs = append(docs, Document{
			Content: currentChunk.String(),
			Source:  fmt.Sprintf("%s (part %d)", path, partNum),
			Metadata: map[string]string{
				"path": path,
				"type": fileType,
				"part": fmt.Sprintf("%d", partNum),
			},
		})
	}

	return docs
}
