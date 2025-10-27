import csv
import os

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    in_path = os.path.join(repo_root, 'data', 'raw', 'diabetes.arff')
    out_path = os.path.join(repo_root, 'data', 'processed', 'diabetes.csv')
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', newline='', encoding='utf-8') as fout:
        headers = []
        data_started = False
        writer = None
        for raw in fin:
            line = raw.strip()
            if not line or line.startswith('%'):
                continue
            low = line.lower()
            if low.startswith('@attribute'):
                # attribute format: @attribute 'name' type  or @attribute name type
                parts = line.split()
                if len(parts) >= 2:
                    attr = parts[1].strip("'\"")
                    headers.append(attr)
            elif low.startswith('@data'):
                data_started = True
                # write header before data
                writer = csv.writer(fout)
                writer.writerow(headers)
            else:
                if not data_started:
                    # ignore any non-attribute lines before @data
                    continue
                vals = [v.strip() for v in line.split(',')]
                writer.writerow(vals)

    print('Wrote CSV to', out_path)

if __name__ == '__main__':
    main()
