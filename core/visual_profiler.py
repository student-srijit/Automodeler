import os
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class VisualProfiler:
    @staticmethod
    def generate_and_upload_eda(bucket, object_key, s3_client, headers, rows):
        print("    -> Visual EDA Generation Started...")
        try:
            # Prepare dataframe
            df = pd.DataFrame(rows, columns=headers)

            # Attempt to convert columns to numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')

            # Identify numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            basename = object_key.split('/')[-1] if '/' in object_key else object_key
            prefix = f"eda-output/{basename}"

            uploaded_files = []

            # 1. Correlation Heatmap (requires at least 2 numeric cols)
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title("Correlation Heatmap")
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                plt.close()

                heatmap_key = f"{prefix}/correlation_heatmap.png"
                s3_client.put_object(Bucket=bucket, Key=heatmap_key, Body=buf.getvalue(), ContentType='image/png')
                uploaded_files.append(heatmap_key)
                print(f"      - Generated: {heatmap_key}")

            # 2. Distribution Histograms for top 5 numeric columns
            for col in numeric_cols[:5]:
                plt.figure(figsize=(8, 6))
                sns.histplot(df[col].dropna(), kde=True, color='skyblue')
                plt.title(f"Distribution of {col}")
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                plt.close()

                hist_key = f"{prefix}/dist_{col}.png"
                s3_client.put_object(Bucket=bucket, Key=hist_key, Body=buf.getvalue(), ContentType='image/png')
                uploaded_files.append(hist_key)
                print(f"      - Generated: {hist_key}")

            print(f"    -> Visual EDA Complete! Uploaded {len(uploaded_files)} plots.")
            return uploaded_files

        except Exception as e:
            print(f"    -> WARNING: Visual EDA failed and was skipped to prevent regression. Error: {e}")
            return []
