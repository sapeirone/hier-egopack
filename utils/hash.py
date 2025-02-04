import hashlib


def compute_hash(file_path, hash_algorithm='sha256'):
    hash_func = hashlib.new(hash_algorithm)
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                hash_func.update(chunk)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    return hash_func.hexdigest()
