import scipy.io
import json
import numpy as np

# Load GolfDB .mat file
mat = scipy.io.loadmat('golfDB.mat')
db = mat['golfDB']  # shape: (1, 1400)

records = []
fields = db.dtype.names

for i in range(db.shape[1]):
    rec = db[0, i]
    item = {}
    for field in fields:
        val = rec[field]
        # Clean numpy structures to clean python types
        if isinstance(val, np.ndarray):
            if val.size == 1:
                val = val.item()
            else:
                val = val.squeeze().tolist()
        item[field] = val
    records.append(item)

# Save to readable JSON
with open('golfDB.json', 'w') as f:
    json.dump(records, f, indent=2)

print(f"Successfully converted {len(records)} GolfDB records to 'golfDB.json'.")

# Print first 3 samples
for r in records[:3]:
    print(r)
