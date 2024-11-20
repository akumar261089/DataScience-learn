# React JS Course Content

## Module 4: Advanced Topics in React

### Lesson 1: Forms and Controlled Components

#### Handling Form Events
- **Theory**: Handling form events involves managing user inputs within a form element, such as text fields, checkboxes, and buttons.
- **Significance**: Capturing and responding to user inputs is essential for interactive applications, especially for login forms, search inputs, and data submission.
- **Usage**: Attach event handlers to form elements and manage form submission.
- **Example**:
  ```javascript
  class MyForm extends React.Component {
    constructor(props) {
      super(props);
      this.state = { value: '' };

      this.handleChange = this.handleChange.bind(this);
      this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleChange(event) {
      this.setState({ value: event.target.value });
    }

    handleSubmit(event) {
      alert('Form submitted with value: ' + this.state.value);
      event.preventDefault();
    }

    render() {
      return (
        <form onSubmit={this.handleSubmit}>
          <label>
            Name:
            <input type="text" value={this.state.value} onChange={this.handleChange} />
          </label>
          <button type="submit">Submit</button>
        </form>
      );
    }
  }
  ```

#### Controlled vs Uncontrolled Components
- **Theory**:
  - **Controlled Components**: Components where form data is handled by the component's state.
  - **Uncontrolled Components**: Components where form data is handled by the DOM itself.
- **Significance**: Controlled components provide greater control over form data, validation, and submission, making them more predictable and easier to debug.
- **Usage**:
  - **Controlled Components**: Use state to manage form values.
    ```javascript
    class ControlledForm extends React.Component {
      constructor(props) {
        super(props);
        this.state = { input: '' };
      }

      handleChange = event => {
        this.setState({ input: event.target.value });
      }

      render() {
        return (
          <input value={this.state.input} onChange={this.handleChange} />
        );
      }
    }
    ```
  - **Uncontrolled Components**: Use `ref` to manage form values.
    ```javascript
    class UncontrolledForm extends React.Component {
      constructor(props) {
        super(props);
        this.input = React.createRef();
      }

      handleSubmit = event => {
        alert('Input value: ' + this.input.current.value);
        event.preventDefault();
      }

      render() {
        return (
          <form onSubmit={this.handleSubmit}>
            <input type="text" ref={this.input} />
            <button type="submit">Submit</button>
          </form>
        );
      }
    }
    ```

#### Using Form Libraries like Formik
- **Theory**: Formik is a popular form management library for React that simplifies handling form state, validation, and submissions.
- **Significance**: It reduces boilerplate code and provides a structured way to handle complex forms with validation and error handling.
- **Usage**: Install Formik and use its components to manage form state and validation.
- **Example**:
  ```javascript
  import React from 'react';
  import { Formik, Field, Form, ErrorMessage } from 'formik';
  import * as Yup from 'yup';

  const SignupForm = () => (
    <Formik
      initialValues={{ email: '', password: '' }}
      validationSchema={Yup.object({
        email: Yup.string().email('Invalid email address').required('Required'),
        password: Yup.string().min(6, 'Must be 6 characters or more').required('Required'),
      })}
      onSubmit={(values, { setSubmitting }) => {
        setTimeout(() => {
          alert(JSON.stringify(values, null, 2));
          setSubmitting(false);
        }, 400);
      }}
    >
      <Form>
        <label htmlFor="email">Email</label>
        <Field name="email" type="email" />
        <ErrorMessage name="email" />

        <label htmlFor="password">Password</label>
        <Field name="password" type="password" />
        <ErrorMessage name="password" />

        <button type="submit">Submit</button>
      </Form>
    </Formik>
  );
  ```

### Lesson 2: Context API

#### Understanding the Need for Context
- **Theory**: Context API allows you to share values (like themes, user information) between components without passing props manually at every level.
- **Significance**: It simplifies state management for deeply nested components and avoids "prop drilling."
- **Usage**: Create and consume context using `React.createContext` and context providers/consumers.
- **Example**:
  ```javascript
  const ThemeContext = React.createContext('light');
  ```

#### Creating Context with React.createContext
- **Theory**: `React.createContext` creates a context object. When React renders a component that subscribes to this context, it will read the current context value from the closest matching `Provider` above it in the tree.
- **Significance**: Provides a way to pass data through the component tree without having to pass props down manually at every level.
- **Usage**:
  ```javascript
  const ThemeContext = React.createContext('light');
  
  class App extends React.Component {
    render() {
      return (
        <ThemeContext.Provider value="dark">
          <Toolbar />
        </ThemeContext.Provider>
      );
    }
  }
  ```

#### Consuming Context with Context.Provider and Context.Consumer
- **Theory**:
  - **Provider**: Provides a context value to its descendants.
  - **Consumer**: Subscribes to context changes and can access the context value.
- **Significance**: Separates context definition (Provider) and usage (Consumer), enabling flexible and maintainable state sharing.
- **Usage**:
  - **Provider**:
    ```javascript
    class App extends React.Component {
      render() {
        return (
          <ThemeContext.Provider value="dark">
            <Toolbar />
          </ThemeContext.Provider>
        );
      }
    }
    ```
  - **Consumer**:
    ```javascript
    function Toolbar() {
      return (
        <ThemeContext.Consumer>
          {value => <h1>Current theme: {value}</h1>}
        </ThemeContext.Consumer>
      );
    }
    ```

### Lesson 3: React Router

#### Setting Up React Router
- **Theory**: React Router is a library for routing in React applications. It enables navigation among views or pages within a single-page application.
- **Significance**: React Router provides a declarative way to manage application navigation, making it easy to handle dynamic routing and nested routes.
- **Usage**: Install React Router, configure routes, and use router components.
- **Example**:
  ```powershell
  npm install react-router-dom
  ```

#### Defining Routes and Navigation
- **Theory**: Routes match URL patterns to components, while navigation links change the URL to rerender the appropriate components.
- **Significance**: Enables user-friendly navigation and deep linking in single-page applications.
- **Usage**:
  ```javascript
  import { BrowserRouter as Router, Route, Link, Switch } from 'react-router-dom';

  function Home() {
    return <h2>Home</h2>;
  }

  function About() {
    return <h2>About</h2>;
  }

  function App() {
    return (
      <Router>
        <div>
          <nav>
            <ul>
              <li>
                <Link to="/">Home</Link>
              </li>
              <li>
                <Link to="/about">About</Link>
              </li>
            </ul>
          </nav>

          <Switch>
            <Route path="/about">
              <About />
            </Route>
            <Route path="/">
              <Home />
            </Route>
          </Switch>
        </div>
      </Router>
    );
  }
  ```

#### Dynamic Routing and Nested Routes
- **Theory**:
  - **Dynamic Routing**: Routes that change based on the URL parameters.
  - **Nested Routes**: Routes that render within other routes, allowing for more complex UI structures.
- **Significance**: Provides fine-grained control over routing logic, enabling nested layouts and dynamic components based on URL parameters.
- **Usage**:
  - **Dynamic Routing**:
    ```javascript
    function User({ match }) {
      return <h2>User ID: {match.params.id}</h2>;
    }

    function App() {
      return (
        <Router>
          <Switch>
            <Route path="/user/:id" component={User} />
          </Switch>
        </Router>
      );
    }
    ```
  - **Nested Routes**:
    ```javascript
    function Category({ match }) {
      return (
        <div>
          <h2>Category: {match.params.category}</h2>
          <Route path={`${match.path}/:subcategory`} component={SubCategory} />
        </div>
      );
    }

    function SubCategory({ match }) {
      return <h3>SubCategory: {match.params.subcategory}</h3>;
    }

    function App() {
      return (
        <Router>
          <Switch>
            <Route path="/category/:category" component={Category} />
          </Switch>
        </Router>
      );
    }
    ```

