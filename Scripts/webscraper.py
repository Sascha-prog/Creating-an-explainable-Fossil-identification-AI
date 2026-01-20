import os
import pandas as pd
import requests
from tqdm import tqdm
import time

def synchronize_dataset(csv_path, output_dir):
    """
    Automated data acquisition utility to synchronize local storage with 
    remote paleontological specimen imagery.
    """
    
    # 1. Establish and verify the local artifact repository
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[System] Created directory: {output_dir}")

    # 2. Load metadata mapping
    # Assumes Column B (Index 1) is 'Filename' and Column C (Index 2) is 'URL'
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Failed to read mapping file: {e}")
        return

    print(f"[System] Metadata loaded. Initializing download for {len(df)} specimens...")

    # 3. Iterative Download Pipeline
    # Using tqdm for progress tracking across the 5,000-specimen corpus
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Downloading Specimens"):
        # Extract metadata based on positional indexing (B=1, C=2)
        # Note: adjust indices if the CSV has a different header structure
        filename = str(row.iloc[1]) 
        image_url = str(row.iloc[2])

        # Ensure filename has an appropriate extension
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename += ".jpg"

        target_path = os.path.join(output_dir, filename)

        # Skip if the specimen is already synchronized locally
        if os.path.exists(target_path):
            continue

        try:
            # Execute stream-based GET request to minimize memory overhead
            response = requests.get(image_url, stream=True, timeout=10)
            
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                print(f"\n[Warning] Failed to fetch {filename}: Status Code {response.status_code}")
                
            # Latency interval to prevent server-side rate-limiting
            time.sleep(0.1) 

        except Exception as e:
            print(f"\n[Error] Exception occurred for specimen {filename}: {e}")

    print(f"\n[System] Dataset synchronization complete. Repository located at: {output_dir}")

if __name__ == "__main__":
    # Define relative paths consistent with the project structure
    CSV_LOCATION = "../Dataset/images_mapping_newest.csv"
    SAVE_DIRECTORY = "../Dataset/images/"
    
    synchronize_dataset(CSV_LOCATION, SAVE_DIRECTORY)