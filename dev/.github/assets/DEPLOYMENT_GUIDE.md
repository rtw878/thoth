# Thoth Brand Deployment Guide

## Overview

This guide documents the complete brand deployment strategy implemented for the Thoth project, following the principles outlined in the Brand Deployment Bible. All brand assets are now centrally managed in the `/.github/assets/` directory.

## Asset Directory Structure

```
/.github/assets/
├── logo.svg                    # Primary logo (SVG)
├── wordmark.svg                # Wordmark only (SVG)
├── favicon.svg                 # SVG favicon with theme support
├── manifest.webmanifest        # PWA manifest
├── palette.md                  # Brand color specifications
├── DEPLOYMENT_GUIDE.md         # This file
└── /png/
    ├── profile-1080x1080.png   # GitHub profile picture
    ├── social-card-1280x640.png # Social media preview
    ├── apple-touch-icon-180x180.png # iOS home screen
    ├── icon-192.png            # PWA icon
    └── icon-512.png            # PWA splash screen
```

## Implementation Summary

### 1. GitHub Repository Branding ✅

#### README.md Theme-Aware Logos
- **Implementation**: HTML `<picture>` element with `prefers-color-scheme` media queries
- **Assets Used**: `logo.svg` and `wordmark.svg`
- **Location**: Lines 1-20 in README.md
- **Benefits**: Crisp rendering on all displays, automatic theme adaptation

#### GitHub Profile Picture
- **Format**: PNG (1080x1080px)
- **Asset**: `/.github/assets/png/profile-1080x1080.png`
- **Usage**: Upload to GitHub organization/user profile settings

#### Social Preview Card
- **Format**: PNG (1280x640px)
- **Asset**: `/.github/assets/png/social-card-1280x640.png`
- **Usage**: GitHub repository settings > Social preview

#### Issue & PR Templates
- **Format**: PNG with full GitHub raw URL
- **Assets**: Branded logos in all templates
- **Benefits**: Consistent branding across GitHub workflows

### 2. Web Application Assets ✅

#### Complete Favicon Set
- **SVG Favicon**: `favicon.svg` - Modern browsers with theme support
- **PNG Icons**: Multiple sizes for PWA and Apple devices
- **PWA Manifest**: `manifest.webmanifest` with icon definitions

#### Theme-Aware Implementation
- **Light Theme**: Primary blue (#1E40AF) on white backgrounds
- **Dark Theme**: Light blue (#60A5FA) on dark backgrounds
- **Accessibility**: WCAG 2.1 AA compliant contrast ratios

### 3. Documentation & Guidelines ✅

#### Brand Color Palette
- **File**: `palette.md`
- **Contents**: Complete color specifications with usage guidelines
- **Accessibility**: Contrast ratios and theme specifications

#### Contributing Documentation
- **Updated**: All references from "Historia Scribe" to "Thoth"
- **Branded**: Added Thoth logo to CONTRIBUTING.md

## Technical Implementation Details

### SVG Optimization
- All SVGs are optimized for web use
- Include proper accessibility attributes (`title`, `desc`, `role="img"`)
- Support CSS manipulation for dynamic theming

### Performance Considerations
- SVG files for primary assets (smaller file sizes)
- PNG fallbacks for compatibility contexts
- Proper asset preloading for critical path

### Accessibility Compliance
- WCAG 2.1 AA contrast ratios
- Screen reader compatible SVG markup
- Semantic HTML structure

## Deployment Checklist

### GitHub Repository
- [ ] Update profile/organization picture with `profile-1080x1080.png`
- [ ] Set social preview with `social-card-1280x640.png`
- [ ] Verify README.md renders correctly in both light and dark themes
- [ ] Test issue and PR template branding

### Web Application
- [ ] Implement favicon set in HTML `<head>`
- [ ] Add PWA manifest reference
- [ ] Test theme switching functionality
- [ ] Verify asset loading performance

### Documentation
- [ ] Review color palette implementation
- [ ] Update any remaining "Historia Scribe" references
- [ ] Verify all asset paths are correct

## Maintenance Guidelines

### Asset Updates
1. **Never modify deployed assets directly**
2. Update master assets in `/.github/assets/`
3. Test all implementations after updates
4. Update this deployment guide if structure changes

### Version Control
- All brand assets are version controlled
- Changes should follow semantic versioning
- Document breaking changes in release notes

## Performance Metrics

### Target KPIs
- **Largest Contentful Paint (LCP)**: < 2.5 seconds
- **Cumulative Layout Shift (CLS)**: < 0.1
- **First Contentful Paint (FCP)**: < 1.8 seconds

### Monitoring
- Regular performance audits
- Cross-browser compatibility testing
- Accessibility compliance checks

## Support & Troubleshooting

### Common Issues
- **SVG not rendering**: Check file paths and MIME types
- **Theme switching not working**: Verify media query syntax
- **PNG blurry on high-DPI**: Ensure high-resolution source files

### Contact
For brand deployment issues, contact: ryan.tris.walmsley@gmail.com

---

*This deployment follows the Brand Deployment Bible & Roadmap for GitHub Repositories v1.0*
