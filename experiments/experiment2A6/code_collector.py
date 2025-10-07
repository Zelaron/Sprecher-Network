import os
import sys
import glob
import shutil
from datetime import datetime
from pathlib import Path

class CodeCollector:
    def __init__(self):
        # Get script directory and name
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.script_name = Path(__file__).name
        
        # Output file path
        self.output_file = self.script_dir / "combined_code.txt"
        
        # Configuration
        self.ignore_empty = True
        self.first_entry = True
        
        # File extensions to process
        self.extensions = [
            "cpp", "cs", "css", "go", "h", "htm", "html", "java", "js", "json",
            "kt", "m", "php", "pl", "py", "rb", "rs", "sh", "svg", "swift", "ts", "xml"
        ]
        
        # Specific files to include regardless of extension
        self.included_files = [
            self.script_dir / "requirements.txt"
        ]
        
        # Directories and files to ignore (using Path objects)
        self.ignored_directories = [
            self.script_dir / "lib",
            self.script_dir / "uploads",
            self.script_dir / "old",
            self.script_dir / "darwin" / "extracted_articles",
            self.script_dir / "venv"
        ]
        
        self.ignored_files = [
            self.script_dir / "directory_tree.py",
            self.script_dir / "from_mnemonic_test.py",
            self.script_dir / "generate_whs.py"
        ]

    def cleanup_directories(self):
        """Clean up __pycache__, venv, .egg-info directories, and .DS_Store files."""
        print("Starting directory cleanup...")
        
        # Find and remove all __pycache__ directories
        for pycache_dir in self.script_dir.rglob("__pycache__"):
            if pycache_dir.is_dir():
                print(f"Removing {pycache_dir}")
                shutil.rmtree(pycache_dir, ignore_errors=True)

        # Remove venv directory if it exists
        venv_dir = self.script_dir / "venv"
        if venv_dir.is_dir():
            print(f"Removing {venv_dir}")
            shutil.rmtree(venv_dir, ignore_errors=True)

        # Find and remove all .egg-info directories
        for egg_info_dir in self.script_dir.rglob("*.egg-info"):
            if egg_info_dir.is_dir():
                print(f"Removing {egg_info_dir}")
                shutil.rmtree(egg_info_dir, ignore_errors=True)

        # Find and remove all .DS_Store files
        for ds_store in self.script_dir.rglob(".DS_Store"):
            if ds_store.is_file():
                print(f"Removing {ds_store}")
                ds_store.unlink()

        print("Directory cleanup completed.")

    def should_skip_file(self, file_path):
        """Check if file should be skipped based on ignore rules."""
        file_path = Path(file_path)
        
        try:
            # Check ignored directories
            for ignored_dir in self.ignored_directories:
                if str(file_path).startswith(str(ignored_dir)):
                    return True
            
            # Check ignored files
            if file_path in self.ignored_files:
                return True
            
            # Check if it's the script itself or output file
            if file_path == self.script_dir / self.script_name:
                return True
            if file_path == self.output_file:
                return True
                
            # Check if file is empty
            if self.ignore_empty and os.path.getsize(file_path) == 0:
                return True
                
            return False
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return True

    def write_file_content(self, file_path):
        """Write file content to the output file with proper formatting."""
        relative_path = str(Path(file_path).relative_to(self.script_dir)).replace('\\', '/')
        
        with open(self.output_file, 'a', encoding='utf-8') as outfile:
            if self.first_entry:
                outfile.write(f"Files in {self.script_dir}:\n\n")
                self.first_entry = False
            
            # Write file header with dashes
            outfile.write("----------\n")
            outfile.write(f"{relative_path}:\n")
            outfile.write("----------\n\n")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                outfile.write("\n\n")  # Add extra newlines between files
            except UnicodeDecodeError:
                print(f"Warning: Unable to read {file_path} as text file")

    def process_directory(self):
        """Process all matching files in the directory."""
        # Perform cleanup first
        self.cleanup_directories()

        # Remove existing output file
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        print("Starting script execution...")
        
        # Process files with specified extensions
        for ext in self.extensions:
            pattern = f"**/*.{ext}"
            for file_path in self.script_dir.glob(pattern):
                if not self.should_skip_file(file_path):
                    self.write_file_content(file_path)

        # Process specifically included files
        for file_path in self.included_files:
            if file_path.exists() and not self.should_skip_file(file_path):
                self.write_file_content(file_path)

        # Remove trailing empty lines
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                content = f.read().rstrip()
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(content)

        if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
            print(f"Script execution completed.")
            print(f"Output file: {self.output_file}")
        else:
            print("No files were processed.")

def main():
    combiner = CodeCollector()
    combiner.process_directory()

if __name__ == "__main__":
    main()
