class DataTransformer:
    @staticmethod
    def transform_rows(headers, rows, profile_columns, duplicate_indices):
        type_map   = {col["column_name"]: col["inferred_type"] for col in profile_columns}
        median_map = {}
        for col in profile_columns:
            if col["inferred_type"] in ("INT8", "FLOAT8"):
                stats = col.get("numeric_stats", {})
                median_map[col["column_name"]] = stats.get("median", 0)

        clean_rows = [r for i, r in enumerate(rows) if i not in duplicate_indices]
        print(f"STAGE 5: Removed {len(rows) - len(clean_rows)} duplicate rows.")

        transformed = []
        for row in clean_rows:
            if len(row) < len(headers):
                row = row + [''] * (len(headers) - len(row))

            clean_row = {}
            for i, header in enumerate(headers):
                raw   = row[i].strip() if i < len(row) else ''
                dtype = type_map.get(header, "STRING")

                if raw == '':
                    if dtype in ("INT8", "FLOAT8"):
                        clean_row[header] = str(median_map.get(header, 0))
                    elif dtype == "BOOL":
                        clean_row[header] = 'false'
                    elif dtype == "TIMESTAMPTZ":
                        clean_row[header] = '1970-01-01T00:00:00'
                    else:
                        clean_row[header] = 'UNKNOWN'
                else:
                    if dtype in ("INT8", "FLOAT8"):
                        clean_row[header] = raw.replace(',', '')
                    elif dtype == "BOOL":
                        clean_row[header] = 'true' if raw.lower() in ('true', 'yes', '1') else 'false'
                    else:
                        clean_row[header] = raw

            transformed.append(clean_row)

        print(f"STAGE 5: {len(transformed)} rows ready to load.")
        return transformed
