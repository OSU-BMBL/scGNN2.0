
[Seurat](https://satijalab.org/seurat) is an famous R toolkit for single cell genomics. Our program provides an interface for Seurat users. 

## Use Seurat object as input
If now you have a `SeuratObject`, then you can export raw counts into `.csv` file from SeuratObject:
```R
write.table(as.matrix(GetAssayData(object = yourSeuratObject, slot = "counts")), 
        '~/counts.csv', 
        sep = ',', row.names = T, col.names = T, quote = F)
```
Then run scGNN2.0 project from this csv file:
```bash
python scGNN_v2.py --load_seurat_object ~/counts.csv \
    --output_dir your_output_dir
```
