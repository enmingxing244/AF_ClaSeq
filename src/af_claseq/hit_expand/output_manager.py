"""
Output manager for hit expand pipeline.

This module provides the HitExpandOutputManager class that handles
output directory creation, file organization, and result management
for the hit expand pipeline.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class HitExpandOutputManager:
    """Manages output directories and files for the hit expand pipeline."""
    
    def __init__(self, base_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the output manager.
        
        Args:
            base_dir: Base directory for all outputs
            logger: Optional logger instance
        """
        self.base_dir = Path(base_dir)
        self.logger = logger or get_logger(__name__)
        
        # Define output directory structure
        self.directories = {
            'clustering': self.base_dir / '00_clustering',
            'msa_pipeline': self.base_dir / '01_msa_pipeline',
            'batches': self.base_dir / '01_msa_pipeline' / 'batches',
            'hit_expansion': self.base_dir / '01_msa_pipeline' / 'hit_expansion',
            'analysis': self.base_dir / '02_analysis',
            'plots': self.base_dir / 'plots',
            'logs': self.base_dir / 'logs'
        }
        
        self.logger.info(f"Output manager initialized for base directory: {self.base_dir}")
    
    def create_directories(self) -> None:
        """Create all necessary output directories."""
        try:
            for name, dir_path in self.directories.items():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {dir_path}")
            
            self.logger.info("All output directories created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating output directories: {str(e)}", exc_info=True)
            raise
    
    def get_directory(self, name: str) -> Path:
        """
        Get a specific output directory by name.
        
        Args:
            name: Directory name ('clustering', 'msa_pipeline', 'batches', etc.)
            
        Returns:
            Path to the requested directory
            
        Raises:
            KeyError: If directory name is not found
        """
        if name not in self.directories:
            raise KeyError(f"Unknown directory name: {name}")
        
        return self.directories[name]
    
    def save_configuration(self, config: Dict[str, Any], filename: str = "hit_expand_config.json") -> Path:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            filename: Output filename
            
        Returns:
            Path to saved configuration file
        """
        try:
            config_file = self.base_dir / filename
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {config_file}")
            return config_file
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
            raise
    
    def save_results_summary(self, results: Dict[str, Any], filename: str = "results_summary.json") -> Path:
        """
        Save results summary to JSON file.
        
        Args:
            results: Results dictionary
            filename: Output filename
            
        Returns:
            Path to saved results file
        """
        try:
            results_file = self.base_dir / filename
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results summary saved to {results_file}")
            return results_file
            
        except Exception as e:
            self.logger.error(f"Error saving results summary: {str(e)}", exc_info=True)
            raise
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, directory: str = "analysis") -> Path:
        """
        Save pandas DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename (should include .csv extension)
            directory: Directory name to save in
            
        Returns:
            Path to saved CSV file
        """
        try:
            output_dir = self.get_directory(directory)
            output_file = output_dir / filename
            
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"DataFrame saved to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving DataFrame: {str(e)}", exc_info=True)
            raise
    
    def copy_file(self, source: Path, destination_dir: str, new_name: Optional[str] = None) -> Path:
        """
        Copy a file to a managed output directory.
        
        Args:
            source: Source file path
            destination_dir: Destination directory name
            new_name: Optional new filename
            
        Returns:
            Path to copied file
        """
        try:
            dest_dir = self.get_directory(destination_dir)
            dest_file = dest_dir / (new_name or source.name)
            
            shutil.copy2(source, dest_file)
            
            self.logger.info(f"File copied from {source} to {dest_file}")
            return dest_file
            
        except Exception as e:
            self.logger.error(f"Error copying file: {str(e)}", exc_info=True)
            raise
    
    def move_file(self, source: Path, destination_dir: str, new_name: Optional[str] = None) -> Path:
        """
        Move a file to a managed output directory.
        
        Args:
            source: Source file path
            destination_dir: Destination directory name
            new_name: Optional new filename
            
        Returns:
            Path to moved file
        """
        try:
            dest_dir = self.get_directory(destination_dir)
            dest_file = dest_dir / (new_name or source.name)
            
            shutil.move(str(source), str(dest_file))
            
            self.logger.info(f"File moved from {source} to {dest_file}")
            return dest_file
            
        except Exception as e:
            self.logger.error(f"Error moving file: {str(e)}", exc_info=True)
            raise
    
    def create_symlink(self, source: Path, destination_dir: str, link_name: Optional[str] = None) -> Path:
        """
        Create a symbolic link to a file in a managed output directory.
        
        Args:
            source: Source file path
            destination_dir: Destination directory name
            link_name: Optional link name
            
        Returns:
            Path to created symlink
        """
        try:
            dest_dir = self.get_directory(destination_dir)
            link_file = dest_dir / (link_name or source.name)
            
            # Remove existing symlink if it exists
            if link_file.is_symlink():
                link_file.unlink()
            
            link_file.symlink_to(source.resolve())
            
            self.logger.info(f"Symlink created from {source} to {link_file}")
            return link_file
            
        except Exception as e:
            self.logger.error(f"Error creating symlink: {str(e)}", exc_info=True)
            raise
    
    def list_files(self, directory: str, pattern: str = "*") -> List[Path]:
        """
        List files in a managed output directory.
        
        Args:
            directory: Directory name
            pattern: File pattern (glob style)
            
        Returns:
            List of file paths
        """
        try:
            output_dir = self.get_directory(directory)
            files = list(output_dir.glob(pattern))
            
            self.logger.debug(f"Found {len(files)} files in {output_dir} matching pattern '{pattern}'")
            return files
            
        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}", exc_info=True)
            raise
    
    def cleanup_temporary_files(self, directory: str, patterns: List[str]) -> None:
        """
        Clean up temporary files in a directory.
        
        Args:
            directory: Directory name
            patterns: List of file patterns to remove
        """
        try:
            output_dir = self.get_directory(directory)
            
            removed_count = 0
            for pattern in patterns:
                files_to_remove = output_dir.glob(pattern)
                for file_path in files_to_remove:
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            removed_count += 1
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                            removed_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {file_path}: {str(e)}")
            
            self.logger.info(f"Cleaned up {removed_count} temporary files/directories")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary files: {str(e)}", exc_info=True)
    
    def get_directory_size(self, directory: str) -> int:
        """
        Get the total size of a directory in bytes.
        
        Args:
            directory: Directory name
            
        Returns:
            Total size in bytes
        """
        try:
            output_dir = self.get_directory(directory)
            
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(output_dir):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
            
            self.logger.debug(f"Directory {output_dir} size: {total_size} bytes")
            return total_size
            
        except Exception as e:
            self.logger.error(f"Error calculating directory size: {str(e)}", exc_info=True)
            return 0
    
    def create_archive(self, directory: str, archive_name: str, format: str = "zip") -> Path:
        """
        Create an archive of a directory.
        
        Args:
            directory: Directory name to archive
            archive_name: Name of the archive file (without extension)
            format: Archive format ('zip', 'tar', 'gztar', 'bztar', 'xztar')
            
        Returns:
            Path to created archive
        """
        try:
            source_dir = self.get_directory(directory)
            archive_file = self.base_dir / f"{archive_name}.{format.replace('tar', 'tar.gz')}"
            
            # Create archive
            shutil.make_archive(
                str(archive_file.with_suffix('')),
                format,
                str(source_dir.parent),
                str(source_dir.name)
            )
            
            self.logger.info(f"Archive created: {archive_file}")
            return archive_file
            
        except Exception as e:
            self.logger.error(f"Error creating archive: {str(e)}", exc_info=True)
            raise
    
    def generate_file_manifest(self, filename: str = "file_manifest.json") -> Path:
        """
        Generate a manifest of all files in the output directory.
        
        Args:
            filename: Manifest filename
            
        Returns:
            Path to manifest file
        """
        try:
            manifest = {}
            
            for name, dir_path in self.directories.items():
                if dir_path.exists():
                    files = []
                    for root, dirs, filenames in os.walk(dir_path):
                        for filename_item in filenames:
                            file_path = Path(root) / filename_item
                            relative_path = file_path.relative_to(self.base_dir)
                            
                            file_info = {
                                'path': str(relative_path),
                                'size': file_path.stat().st_size,
                                'modified': file_path.stat().st_mtime
                            }
                            files.append(file_info)
                    
                    manifest[name] = {
                        'directory': str(dir_path.relative_to(self.base_dir)),
                        'file_count': len(files),
                        'files': files
                    }
            
            manifest_file = self.base_dir / filename
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            self.logger.info(f"File manifest generated: {manifest_file}")
            return manifest_file
            
        except Exception as e:
            self.logger.error(f"Error generating file manifest: {str(e)}", exc_info=True)
            raise
    
    def validate_outputs(self) -> Dict[str, Any]:
        """
        Validate that expected output files exist.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'missing_files': [],
            'missing_directories': [],
            'directory_sizes': {}
        }
        
        try:
            # Check directories
            for name, dir_path in self.directories.items():
                if not dir_path.exists():
                    validation_results['missing_directories'].append(str(dir_path))
                    validation_results['valid'] = False
                else:
                    validation_results['directory_sizes'][name] = self.get_directory_size(name)
            
            # Check specific expected files
            expected_files = [
                'hit_expand_final_msa.a3m',
                'hit_expand_summary.json'
            ]
            
            for expected_file in expected_files:
                file_path = self.base_dir / expected_file
                if not file_path.exists():
                    validation_results['missing_files'].append(expected_file)
                    validation_results['valid'] = False
            
            if validation_results['valid']:
                self.logger.info("Output validation passed")
            else:
                self.logger.warning("Output validation failed")
                self.logger.warning(f"Missing directories: {validation_results['missing_directories']}")
                self.logger.warning(f"Missing files: {validation_results['missing_files']}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating outputs: {str(e)}", exc_info=True)
            validation_results['valid'] = False
            validation_results['error'] = str(e)
            return validation_results