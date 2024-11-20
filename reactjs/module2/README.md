# React JS Course Content

## Module 2: Components and Props

### Lesson 1: Function and Class Components

#### Creating Function Components
- **Significance**: Function components are simpler and easier to write, especially with the advent of React Hooks, which enable them to manage state and side effects.
- **Usage**: Function components are ideal for presentational components or components that simply take props and render UI.
- **Example**:
  ```javascript
  function Welcome(props) {
    return <h1>Hello, {props.name}</h1>;
  }
  
  // Usage
  <Welcome name="Alice" />
  ```

#### Creating Class Components
- **Significance**: Prior to React Hooks, class components were necessary for managing state and lifecycle methods. They are more verbose but still widely used and understood.
- **Usage**: Class components are used when more complex functionalities, particularly lifecycle methods, are needed.
- **Example**:
  ```javascript
  class Welcome extends React.Component {
    render() {
      return <h1>Hello, {this.props.name}</h1>;
    }
  }
  
  // Usage
  <Welcome name="Bob" />
  ```

#### Comparing Function and Class Components
- **Significance**: Understanding the differences allows you to choose the right component type based on the needs of your application.
- **Usage**: Use function components for most scenarios and class components when dealing with older codebases or specific lifecycle uses.
- **Example**:
  - **Function Component**:
    ```javascript
    function Greeting(props) {
      return <h1>{props.message}</h1>;
    }
    ```
  - **Class Component**:
    ```javascript
    class Greeting extends React.Component {
      render() {
        return <h1>{this.props.message}</h1>;
      }
    }
    ```

### Lesson 2: Props

#### Passing Props to Components
- **Significance**: Props (short for properties) are the way to pass data from parent to child components, enabling dynamic and reusable UI components.
- **Usage**: Props make components configurable and reusable.
- **Example**:
  ```javascript
  function Greeting(props) {
      return <h1>{props.message}</h1>;
  }

  // Usage
  <Greeting message="Hello, World!" />
  ```

#### Props Validation with PropTypes
- **Significance**: PropTypes help catch bugs by enforcing the type of props that components receive, improving code reliability and documentation.
- **Usage**: React provides PropTypes as a way to define and validate the types of props a component should receive.
- **Example**:
  ```javascript
  import PropTypes from 'prop-types';

  function Greeting(props) {
      return <h1>{props.message}</h1>;
  }

  Greeting.propTypes = {
      message: PropTypes.string.isRequired,
  };

  // Usage
  <Greeting message="Hello, World!" />
  ```

#### Default Props
- **Significance**: Default props allow components to have default values for certain props, ensuring the component works even if no value is provided.
- **Usage**: Specify default prop values to avoid undefined errors and provide sensible defaults.
- **Example**:
  ```javascript
  function Greeting(props) {
      return <h1>{props.message}</h1>;
  }

  Greeting.defaultProps = {
      message: 'Hello, Guest!',
  };

  // Usage
  <Greeting />  // Renders: Hello, Guest!
  ```

### Lesson 3: Component Composition

#### Composing Components
- **Significance**: Component composition enables building complex UIs from simpler components, facilitating reuse, and improving code maintainability.
- **Usage**: Create complex interfaces by combining smaller, specialized components.
- **Example**:
  ```javascript
  function App() {
      return (
          <div>
              <Header />
              <Body />
              <Footer />
          </div>
      );
  }
  ```

#### Extracting Components for Reusability
- **Significance**: Breaking down a complex component into smaller, reusable parts improves code readability and ensures DRY (Don't Repeat Yourself) principles.
- **Usage**: Identify parts of a UI that can be reused across different parts of the application and extract them into separate components.
- **Example**:
  ```javascript
  function Avatar(props) {
      return <img src={props.user.avatarUrl} alt={props.user.name} />;
  }

  function UserInfo(props) {
      return (
          <div className="UserInfo">
              <Avatar user={props.user} />
              <div className="UserInfo-name">
                  {props.user.name}
              </div>
          </div>
      );
  }
  ```

#### Understanding `children` Props
- **Significance**: The `children` prop allows components to pass nested elements directly, making it easier to create layout components.
- **Usage**: Use `children` to enable components to wrap other components or elements.
- **Example**:
  ```javascript
  function Container(props) {
      return <div className="container">{props.children}</div>;
  }

  // Usage
  <Container>
      <h1>Title</h1>
      <p>This is a paragraph inside the container.</p>
  </Container>
  ```

