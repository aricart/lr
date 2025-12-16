package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ChangeSet represents files that need to be re-indexed
type ChangeSet struct {
	Added    []string // new files
	Modified []string // changed files
	Deleted  []string // removed files
}

// HasChanges returns true if there are any changes
func (cs *ChangeSet) HasChanges() bool {
	return len(cs.Added) > 0 || len(cs.Modified) > 0 || len(cs.Deleted) > 0
}

// ChangedFiles returns all files that need re-indexing (added + modified)
func (cs *ChangeSet) ChangedFiles() []string {
	result := make([]string, 0, len(cs.Added)+len(cs.Modified))
	result = append(result, cs.Added...)
	result = append(result, cs.Modified...)
	return result
}

// RemovedFiles returns all files whose chunks should be removed (modified + deleted)
func (cs *ChangeSet) RemovedFiles() []string {
	result := make([]string, 0, len(cs.Modified)+len(cs.Deleted))
	result = append(result, cs.Modified...)
	result = append(result, cs.Deleted...)
	return result
}

// getGitHeadCommit returns the current HEAD commit hash
func getGitHeadCommit(repoDir string) (string, error) {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = repoDir
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to get git HEAD: %w", err)
	}
	return strings.TrimSpace(string(output)), nil
}

// isGitRepo checks if the directory is a git repository
func isGitRepo(dir string) bool {
	cmd := exec.Command("git", "rev-parse", "--git-dir")
	cmd.Dir = dir
	err := cmd.Run()
	return err == nil
}

// getGitBehindCount returns how many commits the local branch is behind remote
// returns 0 if up to date or if check fails (e.g., no remote, no network)
func getGitBehindCount(repoDir string) int {
	// first, fetch to update remote refs (silently, don't fail if no network)
	fetchCmd := exec.Command("git", "fetch", "--quiet")
	fetchCmd.Dir = repoDir
	fetchCmd.Run() // ignore errors - might be offline

	// check how many commits behind: git rev-list --count HEAD..@{u}
	cmd := exec.Command("git", "rev-list", "--count", "HEAD..@{u}")
	cmd.Dir = repoDir
	output, err := cmd.Output()
	if err != nil {
		return 0 // no upstream or other error
	}

	var count int
	fmt.Sscanf(strings.TrimSpace(string(output)), "%d", &count)
	return count
}

// detectChangesGit uses git diff to find changed files since last commit
func detectChangesGit(repoDir string, lastCommit string, extensions []string) (*ChangeSet, error) {
	cs := &ChangeSet{}

	if lastCommit == "" {
		return nil, fmt.Errorf("no last commit recorded - full re-index required")
	}

	// get changed files: git diff --name-status <last>..HEAD
	cmd := exec.Command("git", "diff", "--name-status", lastCommit+"..HEAD")
	cmd.Dir = repoDir
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git diff failed: %w", err)
	}

	// parse output: each line is "<status>\t<path>" or "<status>\t<old>\t<new>" for renames
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}

		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			continue
		}

		status := parts[0]
		path := parts[len(parts)-1] // use last part (handles renames)

		// filter by extension
		if !hasMatchingExtension(path, extensions) {
			continue
		}

		switch {
		case strings.HasPrefix(status, "A"): // added
			cs.Added = append(cs.Added, path)
		case strings.HasPrefix(status, "M"): // modified
			cs.Modified = append(cs.Modified, path)
		case strings.HasPrefix(status, "D"): // deleted
			cs.Deleted = append(cs.Deleted, path)
		case strings.HasPrefix(status, "R"): // renamed
			// treat as delete old + add new
			if len(parts) >= 3 {
				oldPath := parts[1]
				if hasMatchingExtension(oldPath, extensions) {
					cs.Deleted = append(cs.Deleted, oldPath)
				}
			}
			cs.Added = append(cs.Added, path)
		}
	}

	return cs, nil
}

// detectChangesMtime compares file mtimes against index timestamp
func detectChangesMtime(rootDir string, indexedAt time.Time, indexedFiles []string, extensions []string) (*ChangeSet, error) {
	cs := &ChangeSet{}

	// build set of previously indexed files
	indexedSet := make(map[string]bool)
	for _, f := range indexedFiles {
		indexedSet[f] = true
	}

	// track which indexed files still exist
	stillExists := make(map[string]bool)

	// walk directory and check mtimes
	err := filepath.WalkDir(rootDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() {
			// skip common directories
			dirName := d.Name()
			if dirName == "node_modules" || dirName == ".git" || dirName == "vendor" ||
				dirName == "dist" || dirName == "build" || dirName == ".github" {
				return filepath.SkipDir
			}
			return nil
		}

		relPath, _ := filepath.Rel(rootDir, path)

		// filter by extension
		if !hasMatchingExtension(relPath, extensions) {
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return nil // skip files we can't stat
		}

		if indexedSet[relPath] {
			stillExists[relPath] = true
			// check if modified since indexing
			if info.ModTime().After(indexedAt) {
				cs.Modified = append(cs.Modified, relPath)
			}
		} else {
			// new file
			cs.Added = append(cs.Added, relPath)
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	// find deleted files
	for f := range indexedSet {
		if !stillExists[f] {
			cs.Deleted = append(cs.Deleted, f)
		}
	}

	return cs, nil
}

// hasMatchingExtension checks if path has one of the given extensions
func hasMatchingExtension(path string, extensions []string) bool {
	for _, ext := range extensions {
		if strings.HasSuffix(strings.ToLower(path), ext) {
			return true
		}
	}
	return false
}

// findExistingIndex finds the most recent index file matching the name pattern
func findExistingIndex(indexDir, name string) (string, error) {
	pattern := filepath.Join(indexDir, name+"_*.lrindex")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return "", err
	}

	if len(matches) == 0 {
		return "", fmt.Errorf("no existing index found matching %s", name)
	}

	// return most recent (they're date-stamped, so alphabetically last)
	var latest string
	for _, m := range matches {
		if m > latest {
			latest = m
		}
	}

	return latest, nil
}

// atomicSave saves to a temp file, validates, then renames to final path
func atomicSave(vs *VectorStore, finalPath string) error {
	// write to temp file (keep .lrindex extension for proper gzip compression)
	var tempPath string
	if strings.HasSuffix(finalPath, ".lrindex") {
		tempPath = strings.Replace(finalPath, ".lrindex", ".tmp.lrindex", 1)
	} else {
		tempPath = finalPath + ".tmp"
	}
	if err := vs.Save(tempPath); err != nil {
		return fmt.Errorf("failed to save temp file: %w", err)
	}

	// validate by loading
	testVs := NewVectorStore()
	if err := testVs.Load(tempPath); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("validation failed - temp file corrupt: %w", err)
	}

	// verify chunk count matches
	if len(testVs.Chunks) != len(vs.Chunks) {
		os.Remove(tempPath)
		return fmt.Errorf("validation failed - chunk count mismatch: got %d, expected %d",
			len(testVs.Chunks), len(vs.Chunks))
	}

	// atomic rename
	if err := os.Rename(tempPath, finalPath); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to rename temp file: %w", err)
	}

	return nil
}
