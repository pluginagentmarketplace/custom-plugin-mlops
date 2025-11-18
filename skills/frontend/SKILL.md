---
name: frontend-development
description: Master modern frontend web development with HTML5, CSS3, JavaScript, TypeScript, React, Vue, Angular, Next.js, UX Design, and Design Systems. Learn responsive design, accessibility, performance optimization, and professional development practices.
---

# Frontend Development Skills

## Quick Start

Frontend development involves creating interactive, user-friendly web interfaces. Start with HTML for structure, CSS for styling, and JavaScript for interactivity.

### Essential HTML5
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Website</title>
</head>
<body>
    <header>
        <nav>
            <a href="#home">Home</a>
            <a href="#about">About</a>
        </nav>
    </header>
    <main>
        <h1>Welcome</h1>
        <p>Content here</p>
    </main>
</body>
</html>
```

### Modern CSS Layout
```css
/* Flexbox */
.container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}

/* CSS Grid */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        flex-direction: column;
    }
}
```

### JavaScript Fundamentals
```javascript
// ES6+ features
const greeting = (name) => `Hello, ${name}!`;

async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}

// DOM manipulation
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.querySelector('.btn');
    btn.addEventListener('click', () => {
        btn.classList.toggle('active');
    });
});
```

### React Component Example
```javascript
import React, { useState, useEffect } from 'react';

function Counter() {
    const [count, setCount] = useState(0);

    useEffect(() => {
        document.title = `Count: ${count}`;
    }, [count]);

    return (
        <div className="counter">
            <p>Count: {count}</p>
            <button onClick={() => setCount(count + 1)}>
                Increment
            </button>
        </div>
    );
}

export default Counter;
```

## Core Topics

### 1. HTML5 & Semantic Markup
- Semantic HTML elements (article, section, nav, header, footer)
- Accessibility best practices (ARIA labels, semantic structure)
- Form validation and input types
- Meta tags and SEO optimization

### 2. CSS3 & Styling
- Layout systems (Flexbox, CSS Grid)
- Responsive design (mobile-first approach, media queries)
- CSS Variables (custom properties)
- Animations and transitions
- Styling methodologies (BEM, CSS Modules, Tailwind)

### 3. JavaScript & ES6+
- Variables (const, let), scope, hoisting
- Functions (arrow functions, closures)
- Objects and arrays (destructuring, spread operator)
- Async/await and Promises
- DOM manipulation and events

### 4. TypeScript
- Type annotations and interfaces
- Generics and type inference
- Enums and union types
- Decorators and advanced types

### 5. React Ecosystem
- Components (functional, hooks)
- State management (useState, useContext, Redux)
- Props and component composition
- Lifecycle and useEffect
- Custom hooks and performance optimization

### 6. Vue.js & Angular
- Template syntax and data binding
- Directives and event handling
- Component lifecycle
- State management (Vuex, NgRx)

### 7. Next.js & Meta-Frameworks
- Server-side rendering (SSR)
- Static site generation (SSG)
- API routes and serverless functions
- Image and font optimization
- Routing and dynamic pages

### 8. Web Accessibility (WCAG)
- Keyboard navigation
- Screen reader compatibility
- Color contrast and visual impairment considerations
- Semantic HTML and ARIA labels
- Testing accessibility

### 9. Performance Optimization
- Core Web Vitals (LCP, FID, CLS)
- Code splitting and lazy loading
- Image optimization
- Bundle analysis and optimization
- Caching strategies

### 10. Testing & Debugging
- Unit testing (Jest, Vitest)
- Integration testing
- End-to-end testing (Cypress, Playwright)
- Browser DevTools
- Error tracking and monitoring

## Learning Path

### Month 1-2: Foundations
- HTML5 semantic elements
- CSS3 fundamentals (Flexbox, Grid)
- JavaScript basics and DOM manipulation
- Git version control

### Month 3-4: Intermediate
- Responsive design and mobile-first
- TypeScript basics
- React fundamentals and hooks
- API integration (Fetch, Axios)

### Month 5-6: Advanced
- State management (Redux, Context API)
- Next.js or other meta-frameworks
- Performance optimization
- Testing frameworks and CI/CD

### Month 7+: Professional
- Design systems and component libraries
- Advanced TypeScript patterns
- Production deployment and monitoring
- Team leadership and code reviews

## Tools & Technologies

### Development
- **Editors**: VS Code, WebStorm, Zed
- **Bundlers**: Vite, Webpack, Esbuild
- **Package Managers**: npm, yarn, pnpm

### Frameworks
- **React**: Hooks, Context API, Next.js
- **Vue**: Composition API, Nuxt
- **Angular**: RxJS, dependency injection

### Testing
- **Unit**: Jest, Vitest
- **Integration**: React Testing Library
- **E2E**: Cypress, Playwright, Puppeteer

### Styling
- **CSS**: Tailwind, CSS Modules, styled-components
- **Preprocessors**: SASS/SCSS, PostCSS

### Build & Deploy
- **CI/CD**: GitHub Actions, GitLab CI
- **Hosting**: Vercel, Netlify, AWS Amplify
- **Monitoring**: Sentry, LogRocket

## Advanced Topics

### Design Systems & Component Libraries
- **Component Documentation**: Storybook for interactive component catalog
- **Design Tokens**: Figma Design Tokens, Token Studio
- **CSS Architecture**: Atomic Design, modular component structure
- **Theming Systems**: Dark mode, multi-theme support, CSS-in-JS theme providers
- **Component Composition**: Compound components, render props, slot patterns
- **Documentation**: Component APIs, usage examples, variations

### Performance Optimization Deep Dive
- **Web Workers**: Offload heavy computations from main thread
- **Service Workers**: Offline support, push notifications, background sync
- **WebAssembly (WASM)**: High-performance computations (e.g., image processing)
- **Tree Shaking**: Eliminate unused code from production builds
- **Critical Rendering Path**: Optimize CSS, JavaScript loading order
- **Resource Hints**: DNS prefetch, preload, prefetch, preconnect
- **Streaming SSR**: Progressive rendering with React 18+
- **Hydration**: Client-side hydration of server-rendered content
- **Memory Leaks**: Event listener cleanup, preventing detached DOMs

### Advanced State Management
- **Zustand**: Lightweight alternative to Redux
- **Jotai**: Primitive atomic state management
- **Recoil**: Facebook's experimental state management
- **XState**: State machines for complex UI logic
- **MobX**: Reactive programming approach
- **Pattern**: Lifting state, context boundaries, memoization

### Real-time & WebSocket Features
- **WebSocket APIs**: Bi-directional communication
- **Server-Sent Events (SSE)**: Server-to-client streaming
- **Socket.io**: Fallback support, real-time updates
- **Operational Transformation**: Collaborative editing
- **CRDT (Conflict-free Replicated Data Type)**: Distributed data structures
- **Real-time Sync Patterns**: Optimistic updates, conflict resolution

### Testing Strategy
- **Test Pyramid**: Unit → Integration → E2E
- **Component Testing**: Testing behavior, not implementation
- **Visual Regression Testing**: Percy, Chromatic
- **Accessibility Testing**: Automated (axe), manual WCAG validation
- **Performance Testing**: Lighthouse, WebPageTest
- **Load Testing**: Simulating user load
- **Snapshot Testing**: Detecting unintended changes (use carefully!)

### Security Best Practices
- **XSS Prevention**: Input sanitization, output encoding, CSP headers
- **CSRF Protection**: Token-based protection, SameSite cookies
- **CORS Security**: Proper CORS configuration
- **Content Security Policy**: Strict CSP headers
- **Dependency Security**: npm audit, Snyk, dependabot
- **Secret Management**: Environment variables, secret rotation
- **API Key Rotation**: Prevent long-lived credentials

### Internationalization (i18n) & Localization (l10n)
- **i18n Libraries**: react-i18next, next-i18next
- **Plural Forms**: Handling singular/plural variations
- **Date/Time Formatting**: Locale-specific formatting
- **Right-to-Left (RTL)**: Arabic, Hebrew support
- **Translation Management**: Crowdin, lokalise
- **String Extraction**: Automated translation workflows

### Browser Compatibility & Cross-browser Testing
- **Polyfills**: Babel polyfills for older browsers
- **Feature Detection**: Graceful degradation
- **Testing Services**: BrowserStack, Sauce Labs
- **Caniuse.com**: Checking browser support

## Common Pitfalls & Gotchas

1. **Hydration Mismatch**: Server HTML differs from client render (Next.js, Remix issue)
   - **Fix**: Ensure consistent rendering, use `suppressHydrationWarning` sparingly

2. **React Key Anti-patterns**: Using array index as key causes bugs with dynamic lists
   - **Fix**: Use unique, stable identifiers from data

3. **Closure in Loops**: Variable values captured incorrectly
   ```javascript
   // ❌ Wrong
   for (var i = 0; i < 3; i++) {
     setTimeout(() => console.log(i), 100); // Logs 3, 3, 3
   }
   // ✅ Correct
   for (let i = 0; i < 3; i++) {
     setTimeout(() => console.log(i), 100); // Logs 0, 1, 2
   }
   ```

4. **Missing Dependencies in useEffect**: Causes stale closures and memory leaks
   - **Fix**: Use ESLint `exhaustive-deps` rule

5. **Styling Specificity Wars**: Over-specific selectors cause maintenance issues
   - **Fix**: Use CSS Modules or CSS-in-JS with scoped styles

6. **N+1 Query Problem**: Fetching one item per render
   - **Fix**: Batch requests, use DataLoader pattern

7. **Bundle Size Bloat**: Importing entire libraries when only need a function
   - **Fix**: Tree shake, use code splitting, audit with `webpack-bundle-analyzer`

8. **Accessibility Overlooked**: Adding ARIA without semantic HTML
   - **Fix**: Use semantic HTML first, ARIA when necessary

9. **Mobile Viewport Issues**: Not considering touch targets, tap delays
   - **Fix**: 48px minimum touch targets, remove 300ms tap delay with `viewport-fit`

10. **SEO Neglect in SPAs**: Meta tags not updating dynamically
    - **Fix**: Use Next.js/Nuxt/meta tag libraries for SSR/SSG

## Production Deployment Checklist

- [ ] Bundle size analyzed and optimized (< 100KB gzipped)
- [ ] Core Web Vitals measured (LCP < 2.5s, FID < 100ms, CLS < 0.1)
- [ ] Accessibility audit passed (WCAG AA minimum)
- [ ] Security headers configured (CSP, X-Frame-Options, etc.)
- [ ] Error tracking configured (Sentry, Rollbar)
- [ ] Monitoring & analytics set up
- [ ] CDN configured for static assets
- [ ] Database query performance reviewed
- [ ] Rate limiting implemented for APIs
- [ ] Backup & disaster recovery plan in place

## Best Practices

1. **Semantic HTML** - Use proper elements for accessibility and SEO
2. **Mobile-First** - Design for mobile, scale to desktop
3. **Performance** - Measure, profile, optimize (Real User Monitoring)
4. **Accessibility** - Make sites usable for everyone (WCAG 2.1 AA)
5. **Component Design** - Reusable, testable, documented components
6. **Code Quality** - ESLint, Prettier, TypeScript for type safety
7. **Testing** - Comprehensive unit, integration, and E2E coverage
8. **Documentation** - Clear README, API docs, Storybook, architecture decisions (ADRs)
9. **Security** - Regular dependency updates, security scanning, secure coding practices
10. **Performance Monitoring** - Real User Monitoring (RUM), Synthetic Monitoring

## Architecture Patterns

### Component Architecture
```
components/
├── common/           # Reusable across the app
│   ├── Button.tsx
│   ├── Input.tsx
│   └── Card.tsx
├── features/         # Feature-specific components
│   ├── UserProfile/
│   ├── Dashboard/
│   └── Settings/
└── layouts/          # Page layouts
    ├── MainLayout.tsx
    └── AuthLayout.tsx
```

### Feature Module Structure
```
features/UserAuth/
├── components/       # Feature-specific UI components
├── hooks/            # Custom hooks (useAuth, etc.)
├── context/          # Feature state
├── services/         # API calls
├── types/            # TypeScript interfaces
└── __tests__/        # Feature tests
```

## Performance Optimization Checklist

- [ ] Images optimized (WebP format, lazy loading)
- [ ] Fonts optimized (subset, system fonts, fallbacks)
- [ ] Code splitting by route and component
- [ ] Minification and compression enabled
- [ ] Tree shaking verified
- [ ] Critical CSS inlined
- [ ] Unused CSS removed (PurgeCSS/TailwindCSS)
- [ ] Database query optimized (N+1 queries fixed)
- [ ] Caching strategy implemented (HTTP cache headers, CDN)
- [ ] Monitoring alerts set for performance degradation

## Testing Best Practices

```javascript
// Good test - tests behavior, not implementation
test('submit button is disabled when form has errors', () => {
  render(<LoginForm />);
  const input = screen.getByLabelText('Email');

  userEvent.type(input, 'invalid-email');
  const submitBtn = screen.getByRole('button', { name: /submit/i });

  expect(submitBtn).toBeDisabled();
});

// Bad test - tests implementation detail
test('setIsSubmitting is called', () => {
  const setIsSubmitting = jest.fn();
  render(<LoginForm setIsSubmitting={setIsSubmitting} />);
  // Tests internal state, not user experience
});
```

## Resources & Learning

### Documentation
- [MDN Web Docs](https://developer.mozilla.org) - Authoritative web standards reference
- [Web.dev](https://web.dev) - Google's web development guide with codelabs
- [Can I Use](https://caniuse.com) - Browser feature support
- [HTML Living Standard](https://html.spec.whatwg.org/)

### Frameworks & Libraries
- [React Documentation](https://react.dev)
- [Vue.js Guide](https://vuejs.org/guide)
- [Angular Docs](https://angular.io/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com)

### Learning Platforms
- [Frontend Masters](https://frontendmasters.com) - In-depth courses
- [Egghead.io](https://egghead.io) - Bite-sized courses
- [Scrimba](https://scrimba.com) - Interactive learning
- [CSS-Tricks](https://css-tricks.com) - Deep dives into CSS

### Keeping Up to Date
- [JavaScript Weekly](https://javascriptweekly.com)
- [React Status](https://react.statuspage.io)
- [CSS Weekly](https://css-weekly.com)
- [Web Development Conferences](https://www.smashingconf.com)

## Interview Preparation

### Common Frontend Interview Topics
1. **React Hooks Deep Dive**: useEffect, useState, useCallback, useMemo
2. **Event Loop & Async JavaScript**: Macrotasks, microtasks, event loop visualization
3. **CSS Specificity & Cascade**: Calculating specificity, inheritance, the cascade
4. **DOM API**: querySelector, createElement, event delegation
5. **TypeScript Generics**: Writing reusable, type-safe components
6. **Performance Optimization**: Critical rendering path, jank prevention
7. **State Management**: When to use Redux, Context API, or simpler solutions
8. **Testing Strategy**: What to test, unit vs. integration vs. E2E

### System Design Interview
- **Scalability**: How would you scale a web app to millions of users?
- **Performance**: Design a system for 1M+ concurrent users
- **Architecture**: Monolith vs. Micro frontends trade-offs
- **Case Studies**: Design Twitter, Netflix UI, etc.

## Next Steps

1. **Master your chosen framework** - React/Vue/Angular expertise
2. **Build production projects** - Real-world problems, not tutorials
3. **Contribute to open source** - Learn from experienced developers
4. **Study performance** - Use Lighthouse, WebPageTest, browser DevTools
5. **Focus on accessibility** - Build inclusive experiences from the start
6. **Stay current** - Follow web standards evolution, new APIs (View Transitions, Signals, etc.)
7. **Deep dive on weakness** - If weak on CSS, spend time on it; TypeScript not confident, master it
