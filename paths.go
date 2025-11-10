package main

import (
	"os"
	"path/filepath"
)

// getDataDir returns the directory for storing indexes
// follows XDG base directory specification
func getDataDir() string {
	// check XDG_DATA_HOME first
	if dataHome := os.Getenv("XDG_DATA_HOME"); dataHome != "" {
		return filepath.Join(dataHome, "lr", "indexes")
	}

	// fall back to ~/.local/share/lr/indexes
	home, err := os.UserHomeDir()
	if err != nil {
		// fallback to current directory if home not found
		return "indexes"
	}

	return filepath.Join(home, ".local", "share", "lr", "indexes")
}

// getConfigDir returns the directory for storing configuration
// follows XDG base directory specification
func getConfigDir() string {
	// check XDG_CONFIG_HOME first
	if configHome := os.Getenv("XDG_CONFIG_HOME"); configHome != "" {
		return filepath.Join(configHome, "lr")
	}

	// fall back to ~/.config/lr
	home, err := os.UserHomeDir()
	if err != nil {
		// fallback to current directory if home not found
		return "."
	}

	return filepath.Join(home, ".config", "lr")
}

// ensureDir creates a directory if it doesn't exist
func ensureDir(path string) error {
	return os.MkdirAll(path, 0755)
}

// getDefaultIndexDir returns the default directory for indexes
func getDefaultIndexDir() string {
	dir := getDataDir()
	ensureDir(dir) // create if doesn't exist
	return dir
}

// getEnvFilePath returns the path to the .env file
func getEnvFilePath() string {
	// check in order: current dir, config dir
	if _, err := os.Stat(".env"); err == nil {
		return ".env"
	}

	configDir := getConfigDir()
	return filepath.Join(configDir, "env")
}
