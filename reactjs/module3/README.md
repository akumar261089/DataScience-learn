# React JS Course Content

## Module 3: State and Lifecycle

### Lesson 1: State in React

#### Adding State to a Class Component
- **Theory**: State is an object that determines the behavior and rendering of a component. In class components, state is managed through `this.state` and `this.setState`.
- **Significance**: Managing state allows components to create interactive and dynamic UIs. It keeps track of data that changes over time, like form inputs or a user's actions.
- **Usage**:
  1. Initialize state in the constructor.
  2. Update state using `this.setState`.
- **Example**:
  ```javascript
  class Counter extends React.Component {
    constructor(props) {
      super(props);
      this.state = { count: 0 };
    }

    increment = () => {
      this.setState({ count: this.state.count + 1 });
    }

    render() {
      return (
        <div>
          <p>Count: {this.state.count}</p>
          <button onClick={this.increment}>Increment</button>
        </div>
      );
    }
  }
  ```

#### Functional Components and the useState Hook
- **Theory**: `useState` is a Hook that lets you add state to functional components.
- **Significance**: `useState` simplifies the management of state in functional components and avoids boilerplate code found in class components.
- **Usage**:
  1. Import `useState` from React.
  2. Use `useState` to declare state variables.
- **Example**:
  ```javascript
  import React, { useState } from 'react';

  function Counter() {
    const [count, setCount] = useState(0);

    return (
      <div>
        <p>Count: {count}</p>
        <button onClick={() => setCount(count + 1)}>Increment</button>
      </div>
    );
  }
  ```

#### Updating State
- **Theory**: In both class and functional components, state updates trigger re-renders. State updates can be synchronous (in class components) or asynchronous (in functional components).
- **Significance**: Updating state correctly is crucial for ensuring that the UI reflects the current application state and prevents bugs or inconsistent behavior.
- **Usage**: 
  - **Class Component**: Using `this.setState`.
    ```javascript
    increment = () => {
      this.setState((prevState) => ({ count: prevState.count + 1 }));
    }
    ```
  - **Functional Component**: Using `setState`.
    ```javascript
    const increment = () => {
      setCount((prevCount) => prevCount + 1);
    }
    ```

### Lesson 2: Lifecycle Methods

#### Mounting, Updating, and Unmounting Phases
- **Theory**:
  - **Mounting**: When a component is being inserted into the DOM.
  - **Updating**: When a component is being re-rendered due to changes in props or state.
  - **Unmounting**: When a component is being removed from the DOM.
- **Significance**: Knowing these phases allows developers to control side-effects and optimize performance.
- **Usage**:
  - **Mounting Methods**: `componentDidMount`
  - **Updating Methods**: `componentDidUpdate`
  - **Unmounting Methods**: `componentWillUnmount`

#### Using Lifecycle Methods in Class Components
- **Theory**: Lifecycle methods are special methods in class components that run at specific points in a component's life (mount, update, unmount).
- **Significance**: These methods allow for side-effect management, fetching data, setting timers, etc.
- **Usage**:
  - `componentDidMount`: Runs after the component outputs to the DOM for the first time.
  - `componentDidUpdate`: Runs after the component updates.
  - `componentWillUnmount`: Runs before the component is removed from the DOM.
- **Example**:
  ```javascript
  class Timer extends React.Component {
    constructor(props) {
      super(props);
      this.state = { seconds: 0 };
    }

    tick = () => {
      this.setState(state => ({
        seconds: state.seconds + 1
      }));
    }

    componentDidMount() {
      this.interval = setInterval(this.tick, 1000);
    }

    componentDidUpdate() {
      console.log(`Seconds: ${this.state.seconds}`);
    }

    componentWillUnmount() {
      clearInterval(this.interval);
    }

    render() {
      return <h1>Seconds: {this.state.seconds}</h1>;
    }
  }
  ```

#### The Effect of useEffect Hook in Functional Components
- **Theory**: `useEffect` is a Hook for managing side effects in functional components, working similarly to lifecycle methods in class components.
- **Significance**: `useEffect` simplifies code and handles side-effects directly within functional components.
- **Usage**:
  - `useEffect` runs after the first render and after every update.
  - Cleanup code can be returned from `useEffect`.
- **Example**:
  ```javascript
  import React, { useState, useEffect } from 'react';

  function Timer() {
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
      const interval = setInterval(() => {
        setSeconds(s => s + 1);
      }, 1000);

      return () => clearInterval(interval);
    }, []);

    return <h1>Seconds: {seconds}</h1>;
  }
  ```

### Lesson 3: Handling Events

#### Adding Event Listeners
- **Theory**: Adding event listeners in React involves using the `onEvent` attribute syntax, which directly maps to standard DOM events.
- **Significance**: Handling user interactions (clicks, form inputs, etc.) is essential for interactive applications.
- **Usage**: Assign event handlers using JSX syntax.
- **Example**:
  ```javascript
  function handleClick() {
    alert('Button was clicked');
  }

  return <button onClick={handleClick}>Click me</button>;
  ```

#### Binding Event Handlers in Class Components
- **Theory**: In class components, methods do not automatically bind `this`, so you must explicitly bind event handlers.
- **Significance**: Ensures the proper context (`this`) when the event handler is called.
- **Usage**: Bind event handlers in the constructor or use public class fields.
- **Example**:
  ```javascript
  class Toggle extends React.Component {
    constructor(props) {
      super(props);
      this.state = { isOn: true };
      this.handleClick = this.handleClick.bind(this);
    }

    handleClick() {
      this.setState(state => ({
        isOn: !state.isOn
      }));
    }

    render() {
      return (
        <button onClick={this.handleClick}>
          {this.state.isOn ? 'ON' : 'OFF'}
        </button>
      );
    }
  }
  ```

#### Handling Events in Functional Components
- **Theory**: Functional components handle events similarly but without needing to bind `this`.
- **Significance**: Simplifies event handling by using inline functions or closures.
- **Usage**: Directly use the event handler or define it within the component.
- **Example**:
  ```javascript
  import React, { useState } from 'react';

  function Toggle() {
    const [isOn, setIsOn] = useState(true);

    function handleClick() {
      setIsOn(!isOn);
    }

    return (
      <button onClick={handleClick}>
        {isOn ? 'ON' : 'OFF'}
      </button>
    );
  }
  ```

