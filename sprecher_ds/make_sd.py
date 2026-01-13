import struct
import os
import sys

# --- CONFIGURATION ---
INPUT_FILENAME = "sn_weights.bin"  # The file you generated
OUTPUT_FILENAME = "sd.img"         # The image melonDS needs
FAT_FILENAME = b"SN_WEIGH"         # Max 8 chars, UPPERCASE
FAT_EXTENSION = b"BIN"             # Max 3 chars, UPPERCASE

def create_fat16_image():
    # 1. Check for input file
    if not os.path.exists(INPUT_FILENAME):
        print(f"ERROR: '{INPUT_FILENAME}' not found!")
        print("Please place this script in the same folder as your weights file.")
        return

    print(f"Reading {INPUT_FILENAME}...")
    with open(INPUT_FILENAME, "rb") as f:
        file_data = f.read()
    
    file_size = len(file_data)
    
    # --- FAT16 CONSTANTS ---
    SECTOR_SIZE = 512
    CLUSTER_SIZE = 4096       # 4KB (standard for small FAT16)
    SECTORS_PER_CLUSTER = CLUSTER_SIZE // SECTOR_SIZE
    RESERVED_SECTORS = 1      # Only the boot sector is reserved
    ROOT_ENTRIES = 512        # Standard number of file entries in root
    TOTAL_SECTORS = 32768     # 16 MB total size (16 * 1024 * 1024 / 512)
    
    # Calculate sizes for the File Allocation Tables (FATs)
    # Each FAT entry is 2 bytes. We need enough to cover all clusters.
    # Total Clusters = ~32768 / 8 = 4096.
    # FAT Size = 4096 * 2 bytes = 8192 bytes = 16 sectors.
    SECTORS_PER_FAT = 16 
    
    # Offsets (where things start in the file)
    FAT1_OFFSET = RESERVED_SECTORS * SECTOR_SIZE
    FAT2_OFFSET = FAT1_OFFSET + (SECTORS_PER_FAT * SECTOR_SIZE)
    ROOT_OFFSET = FAT2_OFFSET + (SECTORS_PER_FAT * SECTOR_SIZE)
    # Root dir size = 512 entries * 32 bytes = 16384 bytes
    ROOT_SIZE_BYTES = ROOT_ENTRIES * 32
    DATA_OFFSET = ROOT_OFFSET + ROOT_SIZE_BYTES
    
    # 2. Create the blank image buffer
    print("Generating 16MB FAT16 Filesystem...")
    image_data = bytearray(TOTAL_SECTORS * SECTOR_SIZE)

    # 3. Write Boot Sector (Sector 0)
    # JMP instruction + OEM Name
    struct.pack_into('<3s8s', image_data, 0, b'\xEB\x3C\x90', b'MSDOS5.0')
    
    # BIOS Parameter Block (BPB)
    struct.pack_into('<HBHBHHBHHHII', image_data, 11,
        SECTOR_SIZE,            # Bytes per sector (512)
        SECTORS_PER_CLUSTER,    # Sectors per cluster (8)
        RESERVED_SECTORS,       # Reserved sectors (1)
        2,                      # Number of FATs (2)
        ROOT_ENTRIES,           # Root entries (512)
        TOTAL_SECTORS,          # Total sectors (small)
        0xF8,                   # Media descriptor (HDD/SD)
        SECTORS_PER_FAT,        # Sectors per FAT (16)
        32,                     # Sectors per track
        64,                     # Number of heads
        0,                      # Hidden sectors
        0                       # Total sectors (large)
    )
    
    # Extended Boot Signature
    struct.pack_into('<B', image_data, 38, 0x29)     # ID
    struct.pack_into('<I', image_data, 39, 0x12345678) # Serial Number
    struct.pack_into('<11s', image_data, 43, b'NO NAME    ') # Label
    struct.pack_into('<8s', image_data, 54, b'FAT16   ')    # FS Type
    struct.pack_into('<H', image_data, 510, 0xAA55)          # Boot Signature

    # 4. Initialize FAT Tables
    # The first two entries in FAT are reserved: F8 FF FF FF
    fat_entry_0_1 = b'\xF8\xFF\xFF\xFF'
    
    # Write to FAT1
    image_data[FAT1_OFFSET : FAT1_OFFSET+4] = fat_entry_0_1
    # Write to FAT2
    image_data[FAT2_OFFSET : FAT2_OFFSET+4] = fat_entry_0_1
    
    # 5. Allocate Clusters for the File
    num_clusters_needed = (file_size + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    
    # We start allocating at Cluster 2 (First available data cluster)
    current_cluster = 2
    
    # Write the chain into the FAT tables
    for i in range(num_clusters_needed):
        # Calculate offset in FAT table for this cluster
        # Each entry is 2 bytes
        entry_offset = current_cluster * 2
        
        # Determine the value to write (pointer to next cluster)
        if i == num_clusters_needed - 1:
            next_val = 0xFFFF # End of File (EOF)
        else:
            next_val = current_cluster + 1
            
        # Write to FAT1
        struct.pack_into('<H', image_data, FAT1_OFFSET + entry_offset, next_val)
        # Write to FAT2
        struct.pack_into('<H', image_data, FAT2_OFFSET + entry_offset, next_val)
        
        current_cluster += 1

    # 6. Create Directory Entry
    print(f"Adding entry: {FAT_FILENAME.decode()}.{FAT_EXTENSION.decode()}")
    
    # Directory Entry Structure (32 bytes):
    # Name(8), Ext(3), Attr(1), Res(10), Time(2), Date(2), StartCluster(2), Size(4)
    # Attr 0x20 = Archive
    
    dir_entry = struct.pack('<8s3sB10sHHHI',
        FAT_FILENAME,       # Filename (padded to 8 bytes)
        FAT_EXTENSION,      # Extension (padded to 3 bytes)
        0x20,               # Attributes
        b'\x00'*10,         # Reserved
        0, 0,               # Time/Date (Zero is fine)
        2,                  # Starting Cluster (Always 2 for first file)
        file_size           # File size in bytes
    )
    
    # Write to the beginning of Root Directory
    image_data[ROOT_OFFSET : ROOT_OFFSET+32] = dir_entry

    # 7. Write File Data
    # Cluster 2 corresponds to the very beginning of the Data Region
    print(f"Writing {file_size} bytes of data...")
    image_data[DATA_OFFSET : DATA_OFFSET+file_size] = file_data

    # 8. Save Image
    with open(OUTPUT_FILENAME, "wb") as f:
        f.write(image_data)
        
    print("-" * 40)
    print(f"SUCCESS! Created {OUTPUT_FILENAME}")
    print("1. Open melonDS -> Config -> Emu Settings -> DLDI")
    print(f"2. Select '{OUTPUT_FILENAME}'")
    print("3. Ensure 'Enable DLDI' is CHECKED")
    print("-" * 40)

if __name__ == "__main__":
    create_fat16_image()