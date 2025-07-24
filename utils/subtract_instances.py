#!/usr/bin/env python3
"""
Subtract instances from test_instance.txt in verified_dataset_ids.txt and output to a new txt file
"""

def subtract_instances():
    # Read verified_dataset_ids.txt
    with open('/home/swebench/SWE-Search/verified_dataset_ids.txt', 'r') as f:
        verified_ids = set(line.strip() for line in f if line.strip())
    
    # Read test_instance.txt
    with open('/home/swebench/SWE-Search/test_instance.txt', 'r') as f:
        test_ids = set(line.strip() for line in f if line.strip())
    
    # Calculate difference set (verified - test)
    remaining_ids = verified_ids - test_ids
    
    # Sort alphabetically
    remaining_ids_sorted = sorted(remaining_ids)
    
    # Output to new file
    output_file = '/home/swebench/SWE-Search/remaining_instances.txt'
    with open(output_file, 'w') as f:
        for instance_id in remaining_ids_sorted:
            f.write(instance_id + '\n')
    
    # Print statistics
    print(f"Original verified instances: {len(verified_ids)}")
    print(f"Test instances: {len(test_ids)}")
    print(f"Remaining instances: {len(remaining_ids)}")
    print(f"Results saved to: {output_file}")
    
    # Verify no duplicates
    overlap = verified_ids & test_ids
    print(f"Overlapping instances: {len(overlap)}")
    if overlap:
        print("Overlapping instances:", sorted(overlap)[:10], "..." if len(overlap) > 10 else "")

if __name__ == "__main__":
    subtract_instances()
