import os
from PIL import Image

data_dir = 'data'
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, file)
            try:
                # Using 'with' ensures the file is closed automatically
                with Image.open(path) as img:
                    img.verify() 
                
                # Re-open to fully load (verify() only checks headers)
                with Image.open(path) as img:
                    img.load()
                    
            except Exception as e:
                print(f"Corrupt file found: {path} | Error: {e}")
                # Now that the 'with' block is finished, the file is closed
                # and you can safely remove it on Windows.
                try:
                    os.remove(path)
                    print(f"Successfully deleted: {path}")
                except PermissionError:
                    print(f"Could not delete {path} - still locked.")