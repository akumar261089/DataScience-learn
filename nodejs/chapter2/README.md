# Course Content: Building a Web Server with Node.js

## Module 2: Building a Web Server with Node.js

### 2.1 Introduction to HTTP and REST APIs
- **HTTP Methods**
  - **Explanation**: HTTP methods are used to perform CRUD (Create, Read, Update, Delete) operations.
  - **Example**: 
    - `GET`: Retrieve data
    - `POST`: Submit data
    - `PUT`: Update data
    - `DELETE`: Delete data
  - **Significance**: Understanding HTTP methods is essential for designing and interacting with web services.

- **REST Principles**
  - **Explanation**: REST (Representational State Transfer) is an architectural style for designing networked applications.
  - **Example**: Using resource-based URLs and stateless requests.
    ```http
    GET /users/1
    POST /users
    ```
  - **Significance**: RESTful APIs are easy to understand and use, and they promote a uniform interface across the web.

- **Creating a Simple HTTP Server Using `http` Module**
  - **Explanation**: Using Node’s built-in `http` module to create a basic web server.
  - **Example**: 
    ```javascript
    const http = require('http');
    const server = http.createServer((req, res) => {
      res.statusCode = 200;
      res.setHeader('Content-Type', 'text/plain');
      res.end('Hello, world!\n');
    });
    server.listen(3000, () => {
      console.log('Server running at http://127.0.0.1:3000/');
    });
    ```
  - **Significance**: Forms the foundation for understanding web server concepts and HTTP communication.

### 2.2 Using Express.js
- **Introduction to Express.js**
  - **Explanation**: Express.js is a minimal and flexible Node.js web application framework.
  - **Example**: Setting up routes and middleware.
  - **Significance**: Simplifies the development of web applications by providing robust features for handling HTTP requests.

- **Setting up an Express Server**
  - **Explanation**: Installing and configuring an Express server.
  - **Example**: 
    ```javascript
    const express = require('express');
    const app = express();
    app.get('/', (req, res) => {
      res.send('Hello, Express!');
    });
    app.listen(3000, () => {
      console.log('Express server running at http://127.0.0.1:3000/');
    });
    ```
  - **Significance**: Quickly sets up a server to manage routing and middleware.

- **Routing in Express**
  - **Explanation**: Defining routes to handle different HTTP requests.
  - **Example**: 
    ```javascript
    app.get('/users', (req, res) => {
      res.send('Get users');
    });
    app.post('/users', (req, res) => {
      res.send('Create user');
    });
    ```
  - **Significance**: Organizes application logic and routes to separate URL endpoints, making the codebase cleaner and more maintainable.

- **Middleware in Express**
  - **Explanation**: Functions that execute during the lifecycle of a request to the server.
  - **Example**: 
    ```javascript
    app.use(express.json()); // Middleware to parse JSON bodies
    app.use((req, res, next) => {
      console.log(`${req.method} ${req.url}`);
      next();
    });
    ```
  - **Significance**: Enables preprocessing of requests, error handling, and adding functionalities such as authentication and logging.

### 2.3 Handling Requests and Responses
- **Parsing Request Data**
  - **Explanation**: Extracting and processing data from incoming requests.
  - **Example**: 
    ```javascript
    app.use(express.urlencoded({ extended: true })); // Middleware to parse URL-encoded data
    app.post('/form', (req, res) => {
      console.log(req.body);
      res.send('Form data received');
    });
    ```
  - **Significance**: Essential for handling data submitted by clients, such as form inputs.

- **Sending Responses**
  - **Explanation**: Sending different types of responses to the client.
  - **Example**: 
    ```javascript
    app.get('/json', (req, res) => {
      res.json({ message: 'Hello, JSON!' });
    });
    app.get('/text', (req, res) => {
      res.send('Hello, Text!');
    });
    ```
  - **Significance**: Allows the server to communicate effectively with clients by returning appropriate response formats (JSON, HTML, etc.).

- **Working with Query Parameters and Bodies**
  - **Explanation**: Accessing data sent in the URL or request body.
  - **Example**: 
    ```javascript
    app.get('/search', (req, res) => {
      const query = req.query.q;
      res.send(`Search query: ${query}`);
    });
    app.post('/login', (req, res) => {
      const { username, password } = req.body;
      res.send(`Logged in as: ${username}`);
    });
    ```
  - **Significance**: Facilitates dynamic handling of request data, making the application interactive and responsive to client needs.

- **Handling File Uploads**
  - **Explanation**: Uploading files to the server using middleware like `multer`.
  - **Example**: 
    ```javascript
    const multer = require('multer');
    const upload = multer({ dest: 'uploads/' });
    app.post('/upload', upload.single('file'), (req, res) => {
      res.send('File uploaded successfully');
    });
    ```
  - **Significance**: Enables the server to receive and process files sent by clients, essential for many web applications like image or document management systems.

### 2.4 Connecting to a Database
- **Introduction to MongoDB**
  - **Explanation**: MongoDB is a NoSQL database that stores data in JSON-like documents.
  - **Example**: Document structure in MongoDB.
  - **Significance**: Flexible data model that makes MongoDB suitable for a wide variety of applications.

- **CRUD Operations with Mongoose**
  - **Explanation**: Using Mongoose, an ODM (Object Data Modeling) library, to interact with MongoDB.
  - **Example**: 
    ```javascript
    const mongoose = require('mongoose');
    mongoose.connect('mongodb://localhost/mydatabase', { useNewUrlParser: true, useUnifiedTopology: true });
    const userSchema = new mongoose.Schema({
      name: String,
      email: String,
    });
    const User = mongoose.model('User', userSchema);
    // Create
    const newUser = new User({ name: 'John Doe', email: 'john@example.com' });
    newUser.save();
    // Read
    User.find({}, (err, users) => console.log(users));
    // Update
    User.updateOne({ name: 'John Doe' }, { email: 'john.doe@example.com' });
    // Delete
    User.deleteOne({ name: 'John Doe' });
    ```
  - **Significance**: Provides a schema-based solution to model application data, ensuring data integrity and simplifying interactions with MongoDB.

- **Using Other Databases (PostgreSQL, MySQL)**
  - **Explanation**: Connecting and interacting with relational databases using libraries like `pg` for PostgreSQL and `mysql` for MySQL.
  - **Example**: 
    - PostgreSQL:
      ```javascript
      const { Client } = require('pg');
      const client = new Client({ connectionString: 'postgresql://user:password@localhost:5432/mydatabase' });
      client.connect();
      client.query('SELECT * FROM users', (err, res) => {
        console.log(res.rows);
        client.end();
      });
      ```
    - MySQL:
      ```javascript
      const mysql = require('mysql');
      const connection = mysql.createConnection({ host: 'localhost', user: 'user', password: 'password', database: 'mydatabase' });
      connection.connect();
      connection.query('SELECT * FROM users', (err, results) => {
        console.log(results);
        connection.end();
      });
      ```
  - **Significance**: Provides flexibility in choosing databases that best fit the needs of the application, enabling the use of advanced features like transactions and complex queries.
