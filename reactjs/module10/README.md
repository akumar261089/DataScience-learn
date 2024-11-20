# React JS Course Content

## Module 10: Modern React Features

### Lesson 1: Hooks API Deep Dive

#### useReducer vs useState
- **Theory**:
  - **useState**: A hook that lets you add state to functional components.
  - **useReducer**: A hook that provides a more complex state management solution by using a reducer function.
- **Significance**:
  - **useState**: Suitable for simple state scenarios.
  - **useReducer**: Ideal for more complex state logic and when state transitions depend on the previous state.
- **Examples**:
  - **useState**:
    ```javascript
    import React, { useState } from 'react';

    function Counter() {
      const [count, setCount] = useState(0);

      return (
        <div>
          <p>You clicked {count} times</p>
          <button onClick={() => setCount(count + 1)}>Click me</button>
        </div>
      );
    }
    ```
  - **useReducer**:
    ```javascript
    import React, { useReducer } from 'react';

    function counterReducer(state, action) {
      switch (action.type) {
        case 'increment':
          return { count: state.count + 1 };
        case 'decrement':
          return { count: state.count - 1 };
        default:
          throw new Error();
      }
    }

    function Counter() {
      const [state, dispatch] = useReducer(counterReducer, { count: 0 });

      return (
        <div>
          <p>Count: {state.count}</p>
          <button onClick={() => dispatch({ type: 'increment' })}>+</button>
          <button onClick={() => dispatch({ type: 'decrement' })}>-</button>
        </div>
      );
    }
    ```

#### Custom Hooks and Reusability
- **Theory**: Custom hooks allow you to extract component logic into reusable functions.
- **Significance**: Promotes code reusability and clean, maintainable code. Allows for shared logic among different components.
- **Usage**: Create custom hooks for common functionalities like data fetching, form handling, etc.
- **Examples**:
  - **Custom Hook**:
    ```javascript
    import { useState, useEffect } from 'react';

    function useFetch(url) {
      const [data, setData] = useState(null);
      const [loading, setLoading] = useState(true);

      useEffect(() => {
        fetch(url)
          .then(response => response.json())
          .then(data => {
            setData(data);
            setLoading(false);
          });
      }, [url]);

      return { data, loading };
    }
    ```
  - **Using Custom Hook**:
    ```javascript
    function App() {
      const { data, loading } = useFetch('https://api.example.com/data');

      if (loading) return <p>Loading...</p>;

      return (
        <div>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
      );
    }
    ```

#### Context API with Hooks
- **Theory**: The Context API provides a way to share data across components without prop drilling. Hooks such as `useContext` make it easier to consume context data.
- **Significance**: Simplifies state management for global data like themes, user information, and settings.
- **Usage**: Use `createContext`, `useContext`, and `Context.Provider` to manage and access shared state.
- **Examples**:
  - **Setting Up Context**:
    ```javascript
    import React, { createContext, useState } from 'react';

    const ThemeContext = createContext();

    function ThemeProvider({ children }) {
      const [theme, setTheme] = useState('light');

      return (
        <ThemeContext.Provider value={{ theme, setTheme }}>
          {children}
        </ThemeContext.Provider>
      );
    }

    export { ThemeContext, ThemeProvider };
    ```
  - **Consuming Context Data**:
    ```javascript
    import React, { useContext } from 'react';
    import { ThemeContext } from './ThemeProvider';

    function ThemedComponent() {
      const { theme, setTheme } = useContext(ThemeContext);

      return (
        <div style={{ background: theme === 'light' ? '#fff' : '#333', color: theme === 'light' ? '#000' : '#fff' }}>
          <p>The current theme is {theme}</p>
          <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>Toggle Theme</button>
        </div>
      );
    }
    ```

### Lesson 2: Concurrent Mode and Suspense

#### Introduction to Concurrent Mode
- **Theory**: Concurrent Mode is an experimental feature that allows React to work on low-priority updates in the background, ensuring a more responsive user interface.
- **Significance**: Enhances user experience by keeping the interface responsive and fluid, even during heavy computations.
- **Usage**: Enable Concurrent Mode to allow React to interrupt and prepare low-priority updates without blocking high-priority interactions.
- **Examples**:
  - **Enabling Concurrent Mode**:
    ```jsx
    import { createRoot } from 'react-dom/client';

    const root = createRoot(document.getElementById('root'));
    root.render(<App />);
    ```

#### Concurrent Rendering and Suspense for Data Fetching
- **Theory**: Concurrent rendering allows React to pause and resume rendering as required, while Suspense provides a declarative way to handle asynchronous dependencies like data fetching.
- **Significance**: Ensures smoother transitions and better performance for data-heavy applications.
- **Usage**: Combine Concurrent Mode and Suspense to manage loading states and async operations.
- **Examples**:
  - **Using Suspense**:
    ```jsx
    import React, { Suspense } from 'react';
    const OtherComponent = React.lazy(() => import('./OtherComponent'));

    function MyComponent() {
      return (
        <Suspense fallback={<div>Loading...</div>}>
          <OtherComponent />
        </Suspense>
      );
    }
    ```

#### Transition API
- **Theory**: The Transition API helps manage state transitions during updates, preventing disruption while interacting with the application.
- **Significance**: Improves user interaction by prioritizing updates and transitions to keep the UI responsive.
- **Usage**: Use `useTransition` to manage transitions between different UI states.
- **Examples**:
  - **Using useTransition**:
    ```javascript
    import React, { useState, useTransition } from 'react';

    function App() {
      const [isPending, startTransition] = useTransition();
      const [count, setCount] = useState(0);

      const handleClick = () => {
        startTransition(() => {
          setCount(c => c + 1);
        });
      };

      return (
        <div>
          {isPending ? "Loading..." : <p>Count: {count}</p>}
          <button onClick={handleClick}>Increment</button>
        </div>
      );
    }
    ```

### Lesson 3: Server Components

#### Understanding Server and Client Components
- **Theory**: Server components are rendered on the server and sent to the client as HTML, reducing client-side JavaScript and improving load times. Client components handle dynamic and interactive parts of the application.
- **Significance**: Enhances performance by shifting more work to the server and reducing the client-side JavaScript burden.
- **Usage**: Use a combination of server and client components to optimize application performance.
- **Example**:
  - **Simple Server Component**:
    ```javascript
    // server-component.js
    export default function ServerComponent() {
      return <div>This content is rendered on the server.</div>;
    }
    ```

#### Advantages of Server Components
- **Theory**: Server components offer several performance benefits, such as faster initial load times, reduced client-side JavaScript, and better SEO.
- **Significance**: Crucial for improving performance, especially for larger applications with complex states.
- **Usage**: Identify parts of the application that are static or can be rendered on the server to leverage server components.
- **Examples**:
  - **Performance Improvements**:
    - Reduced JavaScript execution time on the client.
    - Improved initial render performance and TTI (Time to Interactive).

#### Setting Up a Project with Server Components
- **Theory**: Setting up a project to use server components involves configuring the server-side rendering framework and integrating server and client components.
- **Significance**: Ensures that the application is optimized for performance by leveraging server-side rendering where appropriate.
- **Usage**: Configure the project to handle server components using frameworks like Next.js.
- **Examples**:
  - **Next.js Setup**:
    ```bash
    npx create-next-app@latest my-app
    cd my-app
    npm run dev
    ```
  - **Creating Server Components in Next.js**:
    ```javascript
    // pages/index.js
    import React from 'react';

    export default function Home() {
      return (
        <div>
          <h1>Welcome to my app</h1>
        </div>
      );
    }
    ```


This comprehensive course content for Module 10 provides an in-depth look at modern React features, covering critical topics from hooks API, concurrent mode and Suspense, to server components. Each section includes theoretical concepts, significance, usage, and detailed examples to ensure a robust understanding and practical application of modern features in React applications.