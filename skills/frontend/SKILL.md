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

## Best Practices

1. **Semantic HTML** - Use proper elements for accessibility and SEO
2. **Mobile-First** - Design for mobile, scale to desktop
3. **Performance** - Measure, profile, optimize
4. **Accessibility** - Make sites usable for everyone
5. **Component Design** - Reusable, testable, documented components
6. **Code Quality** - Linting, formatting, testing
7. **Documentation** - Clear README, API docs, Storybook

## Resources

- [MDN Web Docs](https://developer.mozilla.org)
- [Web.dev](https://web.dev)
- [React Documentation](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com)
- [Frontend Masters](https://frontendmasters.com)

## Next Steps

1. Choose your primary framework (React, Vue, or Angular)
2. Build projects to reinforce learning
3. Focus on accessibility and performance
4. Contribute to open-source projects
5. Stay updated with web platform changes
