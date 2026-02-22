Tools





## Split Large Text Blocks

Accepts markdown files with large text blocks, including files with few–if any–newline characters.

### Procedure
* Split markdown file by periods (`.`) and put each 

`sed -i '' 's/\./.\n/g' "filename.md"`