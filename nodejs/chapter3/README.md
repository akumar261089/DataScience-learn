## Module 3: Authentication and Authorization

### **3.1 User Authentication**
#### Introduction to Authentication
- **Definition**: Authentication is the process of verifying the identity of a user or entity.
- **Examples**:
  - Username and Password
  - Multi-Factor Authentication (MFA)
  - Biometric Authentication (fingerprints, facial recognition)
- **Usage**:
  - Ensures that the system interacts with genuine users.
  - Provides a primary defense line against unauthorized access.
- **Significance**:
  - Protects user data and system resources.
  - Establishes trust in the system and its security measures.

#### Implementing JWT-based Authentication
- **Definition**: JWT (JSON Web Token) is a compact, URL-safe means of representing claims to be transferred between two parties.
- **Example**:
  ```javascript
  const jwt = require('jsonwebtoken');
  const user = { id: 1, username: 'john' };
  const token = jwt.sign(user, 'secret_key', { expiresIn: '1h' });
  console.log(token);
  ```
- **Usage**:
  - Allows stateless authentication using tokens.
  - Tokens can be stored in cookies or local storage and sent with each request.
- **Significance**:
  - Reduces the need for server-side sessions.
  - Enhances scalability by offloading authentication logic from the server.

#### Using Passport.js for Authentication
- **Definition**: Passport.js is a middleware for Node.js that simplifies authentication.
- **Example**:
  ```javascript
  const passport = require('passport');
  const LocalStrategy = require('passport-local').Strategy;

  passport.use(new LocalStrategy(
    function(username, password, done) {
      User.findOne({ username: username }, function (err, user) {
        if (err) { return done(err); }
        if (!user) { return done(null, false); }
        if (!user.verifyPassword(password)) { return done(null, false); }
        return done(null, user);
      });
    }
  ));
  ```
- **Usage**:
  - Integrates with various authentication strategies (local, OAuth, OpenID, etc.).
  - Standardizes the process of authentication across different methods.
- **Significance**:
  - Simplifies the implementation of various authentication workflows.
  - Provides a flexible framework to handle multiple auth strategies.

### **3.2 User Authorization**
#### Understanding Roles and Permissions
- **Definition**: Authorization determines what an authenticated user is allowed to do.
- **Examples**:
  - Admin, Editor, Viewer roles in a content management system.
  - Permissions like read, write, delete, and update.
- **Usage**:
  - Grants specific capabilities to users based on their role.
  - Controls access to resources and functionalities.
- **Significance**:
  - Enhances security by enforcing the principle of least privilege.
  - Simplifies user management and resource allocation.

#### Implementing Role-Based Access Control
- **Example**:
  ```javascript
  function checkRole(role) {
    return function(req, res, next) {
      if (req.user && req.user.role === role) {
        next();
      } else {
        res.status(403).send('Forbidden');
      }
    }
  }
  
  app.get('/admin', checkRole('admin'), function(req, res) {
    res.send('Welcome, admin!');
  });
  ```
- **Usage**:
  - Assign roles to users (e.g., admin, user, guest).
  - Protect routes and resources based on user roles.
- **Significance**:
  - Ensures that only authorized users can access sensitive parts of the application.
  - Provides a clear and manageable structure for user permissions.

### **3.3 Security Best Practices**
#### Securing API Endpoints
- **Practice**: Use HTTPS, validate inputs, use proper authentication and authorization.
- **Example**:
  ```javascript
  app.use('/api', passport.authenticate('jwt', { session: false }));
  ```
- **Significance**: Ensures the integrity and confidentiality of data between clients and servers.

#### Storing Passwords Securely
- **Practice**: Use hashing algorithms like bcrypt for storing passwords.
- **Example**:
  ```javascript
  const bcrypt = require('bcrypt');
  bcrypt.hash('password123', 10, function(err, hash) {
    // Store hash in your database
  });
  ```
- **Significance**: Protects passwords from being exposed in case of a data breach.

#### Preventing Common Security Vulnerabilities
- **Examples**:
  - SQL Injection
    - **Prevention**: Use parameterized queries or ORM tools like Sequelize.
    ```javascript
    db.query('SELECT * FROM users WHERE id = ?', [userId], function(err, results) {
      // handle results
    });
    ```
  - Cross-Site Scripting (XSS)
    - **Prevention**: Sanitize user inputs and use security libraries.
    ```javascript
    const escape = require('escape-html');
    app.get('/search', (req, res) => {
      const query = escape(req.query.q);
      // use query to search
    });
    ```
  - Cross-Site Request Forgery (CSRF)
    - **Prevention**: Use anti-CSRF tokens and security libraries.
    ```javascript
    const csrf = require('csurf');
    app.use(csrf());
    app.get('/form', (req, res) => {
      res.render('form', { csrfToken: req.csrfToken() });
    });
    ```
- **Significance**: Protecting the application from these vulnerabilities ensures its long-term security and reliability.