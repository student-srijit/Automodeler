import csv
import re
import hashlib
import statistics
import json

class DataProfiler:
    DATE_PATTERNS = [
        re.compile(r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?'),
        re.compile(r'^\d{2}/\d{2}/\d{4}'),
        re.compile(r'^\d{2}-\d{2}-\d{4}'),
        re.compile(r'^\d{4}/\d{2}/\d{2}'),
    ]

    @staticmethod
    def infer_type(values):
        if not values:
            return "UNKNOWN"
        int_p  = re.compile(r'^-?[\d,]+$')
        flt_p  = re.compile(r'^-?[\d,]+\.\d+$')
        uuid_p = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
        bool_p = re.compile(r'^(true|false|yes|no|1|0)$', re.IGNORECASE)
        sample = values[:30]
        if all(uuid_p.match(v) for v in sample):                              return "UUID"
        if all(any(p.match(v) for p in DataProfiler.DATE_PATTERNS) for v in sample): return "TIMESTAMPTZ"
        if all(bool_p.match(v) for v in sample):                              return "BOOL"
        if all(int_p.match(v.replace(',', '')) for v in sample):              return "INT8"
        if all(flt_p.match(v.replace(',', '')) for v in sample):              return "FLOAT8"
        return "STRING"

    @staticmethod
    def compute_numeric_stats(values):
        try:
            nums = [float(v.replace(',', '')) for v in values if v]
            if not nums:
                return {}
            q1  = statistics.quantiles(nums, n=4)[0]
            q3  = statistics.quantiles(nums, n=4)[2]
            iqr = q3 - q1
            outliers = [v for v in nums if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr]
            return {
                "min":           round(min(nums), 4),
                "max":           round(max(nums), 4),
                "mean":          round(statistics.mean(nums), 4),
                "median":        round(statistics.median(nums), 4),
                "std_dev":       round(statistics.stdev(nums), 4) if len(nums) > 1 else 0,
                "outlier_count": len(outliers)
            }
        except Exception:
            return {}

    @staticmethod
    def detect_fk_hints(columns_data, headers):
        hints = []
        for col_a in headers:
            vals_a = set(v for v in columns_data[col_a] if v)
            if len(vals_a) < 2:
                continue
            for col_b in headers:
                if col_a == col_b:
                    continue
                vals_b = set(v for v in columns_data[col_b] if v)
                if vals_a.issubset(vals_b) and len(vals_a) < len(vals_b) * 0.6:
                    hints.append(f"{col_a} may reference {col_b}")
        return hints

    @staticmethod
    def profile_csv(bucket, key, s3_client):
        response   = s3_client.get_object(Bucket=bucket, Key=key)
        lines      = response['Body'].read().decode('utf-8', errors='replace').splitlines()
        reader     = csv.reader(lines)
        headers    = [h.strip() for h in next(reader)]
        rows       = list(reader)
        total_rows = len(rows)

        if total_rows == 0:
            raise ValueError("CSV is empty")

        row_hashes = [hashlib.md5(','.join(r).encode()).hexdigest() for r in rows]
        seen = {}
        duplicate_indices = set()
        for i, h in enumerate(row_hashes):
            if h in seen:
                duplicate_indices.add(i)
            else:
                seen[h] = i

        columns_data = {h: [] for h in headers}
        for row in rows:
            for idx, val in enumerate(row):
                if idx < len(headers):
                    columns_data[headers[idx]].append(val.strip())

        profile_columns = []
        for col, values in columns_data.items():
            non_empty     = [v for v in values if v != '']
            unique_vals   = set(non_empty)
            uniqueness_ratio = len(unique_vals) / total_rows if total_rows > 0 else 0
            inferred_type = DataProfiler.infer_type(non_empty)
            is_pk         = (uniqueness_ratio == 1.0) and (len(non_empty) == total_rows)

            col_profile = {
                "column_name":      col,
                "inferred_type":    inferred_type,
                "uniqueness_ratio": round(uniqueness_ratio, 4),
                "null_count":       total_rows - len(non_empty),
                "cardinality":      len(unique_vals),
                "is_pk_candidate":  is_pk,
                "sample_values":    list(unique_vals)[:5],
            }
            if inferred_type in ("INT8", "FLOAT8"):
                col_profile["numeric_stats"] = DataProfiler.compute_numeric_stats(non_empty)

            profile_columns.append(col_profile)

        table_name = key.split('/')[-1].split('.')[0]
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name).lower()
        if not table_name:
            table_name = "submission"

        profile = {
            "table_name":            table_name,
            "total_rows":            total_rows,
            "duplicate_rows":        len(duplicate_indices),
            "columns":               profile_columns,
            "fk_relationship_hints": DataProfiler.detect_fk_hints(columns_data, headers)
        }
        print("STAGE 1 PROFILE:", json.dumps(profile, indent=2))
        return profile, headers, rows, duplicate_indices
