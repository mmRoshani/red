# Red Documentation

This directory contains the complete documentation for the Red Federated Learning Framework, designed for deployment as a GitHub Pages blog-style documentation site.

## 📁 Structure

```
docs/
├── _config.yml           # Jekyll configuration
├── index.md              # Homepage
├── Gemfile               # Jekyll dependencies
├── README.md             # This file
└── _posts/               # Blog-style documentation posts
    ├── 2024-01-01-federated-base.md
    ├── 2024-01-02-virtual-nodes.md
    ├── 2024-01-03-topology-manager.md
    ├── 2024-01-04-federated-node.md
    ├── 2024-01-05-getting-started.md
    ├── 2024-01-06-ray-integration-deep-dive.md
    ├── 2024-01-07-architecture-overview.md
    └── 2024-01-08-api-reference.md
```

## 🚀 Quick Start

### Local Development

1. **Install Jekyll Dependencies**
   ```bash
   cd docs
   bundle install
   ```

2. **Serve Locally**
   ```bash
   bundle exec jekyll serve
   ```

3. **Access Documentation**
   Open `http://localhost:4000` in your browser

### GitHub Pages Deployment

1. **Enable GitHub Pages**
   - Go to repository Settings
   - Navigate to Pages section
   - Set source to "Deploy from a branch"
   - Select `main` branch and `/docs` folder

2. **Access Published Site**
   - Documentation will be available at `https://your-username.github.io/red`

## 📚 Documentation Content

### Core Architecture Documentation

- **[FederatedBase](2024-01-01-federated-base.md)**: Foundation class for all federation implementations
- **[Virtual Nodes](2024-01-02-virtual-nodes.md)**: Lazy initialization and resource management
- **[Topology Manager](2024-01-03-topology-manager.md)**: Communication infrastructure
- **[Federated Node](2024-01-04-federated-node.md)**: Individual participant nodes

### User Guides

- **[Getting Started](2024-01-05-getting-started.md)**: Complete quickstart guide
- **[Ray Integration Deep Dive](2024-01-06-ray-integration-deep-dive.md)**: Advanced Ray usage
- **[Architecture Overview](2024-01-07-architecture-overview.md)**: System design and patterns

### Reference

- **[API Reference](2024-01-08-api-reference.md)**: Complete API documentation

## 🎨 Customization

### Theme

The documentation uses the Minima theme with custom navigation. You can customize:

- **Colors**: Modify `_config.yml` theme settings
- **Navigation**: Update `nav_links` in `_config.yml`
- **Layout**: Customize `_layouts/` directory (if created)

### Adding Content

1. **New Documentation Page**
   ```bash
   # Create new post with date-based naming
   touch _posts/YYYY-MM-DD-your-topic.md
   ```

2. **Post Format**
   ```markdown
   ---
   layout: post
   title: "Your Title"
   date: YYYY-MM-DD HH:MM:SS +0000
   categories: [category1, category2]
   tags: [tag1, tag2, tag3]
   author: Red Team
   ---

   # Your Content Here
   ```

3. **Update Navigation**
   - Add link to `_config.yml` nav_links if needed
   - Reference from other pages using Jekyll post_url

## 🔧 Jekyll Configuration

Key configuration options in `_config.yml`:

```yaml
title: Red Federated Learning Framework
description: A Production-Grade Federated Learning Framework with Ray
author: Red Project Team
url: ""
baseurl: ""

# Build settings
markdown: kramdown
highlighter: rouge
theme: minima

# Plugins
plugins:
  - jekyll-feed
  - jekyll-sitemap
```

## 📖 Content Guidelines

### Writing Style

- **Clear and Concise**: Focus on practical implementation
- **Code Examples**: Include working code snippets
- **Visual Elements**: Use emojis and formatting for readability
- **Cross-References**: Link between related documentation

### Code Formatting

- Use syntax highlighting: ```python
- Include complete, runnable examples
- Add comments explaining key concepts
- Show both basic and advanced usage patterns

### Structure

- **Headers**: Use descriptive H2/H3 headers
- **Code Blocks**: Label with language and context
- **Lists**: Use for step-by-step instructions
- **Tables**: For comparison and reference data

## 🚀 Deployment

### GitHub Actions (Optional)

You can set up automated deployment with GitHub Actions:

```yaml
# .github/workflows/deploy-docs.yml
name: Deploy Documentation

on:
  push:
    branches: [ main ]
    paths: [ 'docs/**' ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Ruby
      uses: ruby/setup-ruby@v1
      with:
        ruby-version: 3.0
        bundler-cache: true
        working-directory: docs
    - name: Build site
      run: bundle exec jekyll build
      working-directory: docs
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_site
```

### Custom Domain (Optional)

To use a custom domain:

1. Add `CNAME` file to docs folder:
   ```
   your-domain.com
   ```

2. Configure DNS records:
   ```
   Type: CNAME
   Name: your-domain.com
   Value: your-username.github.io
   ```

## 🔍 SEO and Analytics

### Search Engine Optimization

The documentation includes:
- **Jekyll SEO Tag**: Automatic meta tags
- **Sitemap**: Generated sitemap.xml
- **Feed**: RSS/Atom feed for updates

### Analytics Integration

Add to `_config.yml`:
```yaml
google_analytics: YOUR_TRACKING_ID
```

## 🤝 Contributing

### Adding Documentation

1. Fork the repository
2. Create documentation in `docs/_posts/`
3. Test locally with `bundle exec jekyll serve`
4. Submit pull request

### Documentation Standards

- Follow existing naming conventions
- Include practical examples
- Test all code snippets
- Update navigation if needed

## 📞 Support

For documentation issues:
- Open GitHub issue with `documentation` label
- Include specific page and section
- Suggest improvements or corrections

---

This documentation system provides a comprehensive, searchable, and maintainable resource for Red users and developers. The blog-style format makes it easy to add new content while maintaining organization and discoverability. 