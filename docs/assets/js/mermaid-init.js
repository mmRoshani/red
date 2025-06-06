// Mermaid initialization for Red framework documentation
document.addEventListener('DOMContentLoaded', function() {
  // Load Mermaid from CDN
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js';
  script.onload = function() {
    // Initialize Mermaid
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      themeVariables: {
        primaryColor: '#0366d6',
        primaryTextColor: '#ffffff',
        primaryBorderColor: '#0366d6',
        lineColor: '#333333',
        sectionBkgColor: '#f6f8fa',
        altSectionBkgColor: '#ffffff',
        gridColor: '#e1e4e8',
        secondaryColor: '#f1f8ff',
        tertiaryColor: '#fff5b4'
      },
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'cardinal'
      },
      sequence: {
        useMaxWidth: true
      },
      gantt: {
        useMaxWidth: true
      }
    });

    // Find all code blocks with language-mermaid class
    const mermaidBlocks = document.querySelectorAll('code.language-mermaid');
    
    mermaidBlocks.forEach((block, index) => {
      // Create a new div for the mermaid diagram
      const mermaidDiv = document.createElement('div');
      mermaidDiv.className = 'mermaid';
      mermaidDiv.id = 'mermaid-' + index;
      mermaidDiv.innerHTML = block.textContent;
      
      // Replace the code block with the mermaid div
      const pre = block.parentElement;
      pre.parentNode.insertBefore(mermaidDiv, pre);
      pre.remove();
    });

    // Also check for pre.mermaid (alternative format)
    const preMermaidBlocks = document.querySelectorAll('pre.mermaid code');
    preMermaidBlocks.forEach((block, index) => {
      const mermaidDiv = document.createElement('div');
      mermaidDiv.className = 'mermaid';
      mermaidDiv.id = 'mermaid-pre-' + index;
      mermaidDiv.innerHTML = block.textContent;
      
      const pre = block.parentElement;
      pre.parentNode.insertBefore(mermaidDiv, pre);
      pre.remove();
    });

    // Initialize mermaid rendering
    mermaid.init(undefined, document.querySelectorAll('.mermaid'));
  };
  
  document.head.appendChild(script);
});

// Add CSS for mermaid diagrams
const style = document.createElement('style');
style.textContent = `
  .mermaid {
    text-align: center;
    margin: 2rem 0;
    background: transparent;
  }
  
  .mermaid svg {
    max-width: 100%;
    height: auto;
  }
`;
document.head.appendChild(style); 