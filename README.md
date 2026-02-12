# knowledge_base_public

Links to other projects: 
* Markdown Viewers:
  * https://github.com/retext-project/retext
    - **Native GUI:** It runs as a standalone desktop window, not a local web server/browser tab.
    - **Live Preview:** It features a synchronized preview pane that updates as the content changes.
    - **Extensible:** Because it is written in Python, you can easily add extensions (like MathJax for formulas).
  * https://github.com/alexaldearroyo/MarkdownViewer
    - **Description:** A compact viewer built specifically with Python and PyQt5 for macOS.
    - **Pros:** Designed solely for viewing; simpler interface.
    - **Cons:** Less actively maintained than ReText.
  * https://github.com/joeyespo/grip
    - The most accurate viewer is Grip, which uses the actual GitHub API to render markdown, ensuring it looks exactly like it does on GitHub.
    - **How it works:** You run grip myfile.md in the terminal, and it opens a browser window that auto-refreshes whenever you save the file.
