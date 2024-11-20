# React JS Course Content

## Module 6: API Integration

### Lesson 1: Fetching Data

#### Using Fetch API to Fetch Data
- **Theory**: The Fetch API provides a simple interface for making HTTP requests in JavaScript. It returns a Promise that resolves to the Response object representing the response to a request.
- **Significance**: Fetch API is built into modern browsers, making it a reliable and readily available choice for handling network requests.
- **Usage**: Use the `fetch` function to make requests to an API endpoint and handle the response.
- **Example**:
  ```javascript
  useEffect(() => {
    fetch('https://api.example.com/data')
      .then(response => response.json())
      .then(data => console.log(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []);
  ```

#### Handling Loading and Error States
- **Theory**: Handling loading and error states involves providing feedback to users while data is being fetched and managing any errors that occur during the request.
- **Significance**: Ensures a better user experience by informing users of the current status of their request and any issues that prevent successful data retrieval.
- **Usage**: Use state variables to track loading and error conditions, and display appropriate messages or indicators.
- **Example**:
  ```javascript
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('https://api.example.com/data')
      .then(response => response.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(error => {
        setError(error);
        setLoading(false);
      });
  }, []);

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return <div>{JSON.stringify(data)}</div>;
  ```

#### Displaying Fetched Data
- **Theory**: Once data is fetched, it needs to be rendered within the UI of the application.
- **Significance**: Displaying data effectively is crucial for user interaction and information presentation.
- **Usage**: Map over the fetched data and display it using appropriate UI components.
- **Example**:
  ```javascript
  return (
    <ul>
      {data.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
  ```

### Lesson 2: Axios and Other Libraries

#### Setting Up Axios
- **Theory**: Axios is a popular HTTP client library for making requests in JavaScript. It provides a simple and flexible API for performing asynchronous HTTP requests.
- **Significance**: Axios simplifies error handling, request/response transformations, and allows for interceptors. It's widely used in the industry due to its ease of use and flexibility.
- **Usage**:
  ```sh
  npm install axios
  ```
  ```javascript
  import axios from 'axios';

  axios.get('https://api.example.com/data')
    .then(response => console.log(response.data))
    .catch(error => console.error('Error fetching data:', error));
  ```

#### Making GET, POST, PUT, DELETE Requests
- **Theory**: HTTP methods such as GET, POST, PUT, DELETE are used to request data, create new data, update existing data, and delete data, respectively.
- **Significance**: Understanding and using these methods correctly is essential for interacting with RESTful APIs and manipulating data.
- **Usage**: Use corresponding Axios methods to perform different types of requests.
- **Example** (GET):
  ```javascript
  axios.get('https://api.example.com/data')
    .then(response => console.log(response.data))
    .catch(error => console.error('Error fetching data:', error));
  ```
  - **Example** (POST):
    ```javascript
    axios.post('https://api.example.com/data', { name: 'New Item' })
      .then(response => console.log(response.data))
      .catch(error => console.error('Error posting data:', error));
    ```
  - **Example** (PUT):
    ```javascript
    axios.put('https://api.example.com/data/1', { name: 'Updated Item' })
      .then(response => console.log(response.data))
      .catch(error => console.error('Error updating data:', error));
    ```
  - **Example** (DELETE):
    ```javascript
    axios.delete('https://api.example.com/data/1')
      .then(response => console.log(response.data))
      .catch(error => console.error('Error deleting data:', error));
    ```

#### Handling Responses and Errors
- **Theory**: Handling responses involves processing the returned data, and managing errors involves catching and responding to any issues that arise during the request.
- **Significance**: Properly handling responses and errors ensures robust and reliable interactions with APIs.
- **Usage**: Use `then` for responses and `catch` for errors.
- **Example**:
  ```javascript
  axios.get('https://api.example.com/data')
    .then(response => {
      console.log('Data fetched successfully:', response.data);
    })
    .catch(error => {
      console.error('Error fetching data:', error);
    });
  ```

### Lesson 3: Advanced Data Fetching

#### Using SWR/react-query for Data Fetching
- **Theory**: Libraries like SWR and react-query provide advanced hooks for data fetching, caching, and synchronization in React applications.
- **Significance**: They simplify complex data-fetching scenarios, improve performance with caching, and ensure data freshness.
- **Usage**: Install the library and use hooks provided for data fetching.
  ```sh
  npm install swr
  ```
  ```sh
  npm install react-query
  ```
  - **Example** (SWR):
    ```javascript
    import useSWR from 'swr';

    const fetcher = url => fetch(url).then(res => res.json());

    function Component() {
      const { data, error } = useSWR('https://api.example.com/data', fetcher);

      if (error) return <div>Failed to load</div>;
      if (!data) return <div>Loading...</div>;

      return <div>{JSON.stringify(data)}</div>;
    }
    ```
  - **Example** (react-query):
    ```javascript
    import { useQuery } from 'react-query';
    import axios from 'axios';

    const fetchData = async () => {
      const { data } = await axios.get('https://api.example.com/data');
      return data;
    };

    function Component() {
      const { data, error, isLoading } = useQuery('fetchData', fetchData);

      if (isLoading) return <div>Loading...</div>;
      if (error) return <div>Error: {error.message}</div>;

      return <div>{JSON.stringify(data)}</div>;
    }
    ```

#### Optimistic Updates
- **Theory**: Optimistic updates immediately update the UI to reflect a mutation, assuming it will succeed, and later corrects the data if the request fails.
- **Significance**: Enhances user experience by making the UI more responsive and reducing the perceived latency.
- **Usage**: Update the state immediately, and then make the API call, rolling back changes if the call fails.
  - **Example** (react-query):
    ```javascript
    import { useMutation, queryClient } from 'react-query';

    const updateData = async (newData) => {
      const { data } = await axios.post('https://api.example.com/data', newData);
      return data;
    };

    function Component() {
      const mutation = useMutation(updateData, {
        onMutate: newData => {
          // Optimistically update the UI
          queryClient.setQueryData('fetchData', oldData => [...oldData, newData]);
        },
        onError: (error, newData, rollback) => {
          // Rollback changes if error occurs
          queryClient.setQueryData('fetchData', rollback);
        },
        onSettled: () => {
          // Refetch the data
          queryClient.invalidateQueries('fetchData');
        }
      });

      const handleAdd = (newItem) => {
        mutation.mutate(newItem);
      };

      return <button onClick={() => handleAdd({ name: 'New Item' })}>Add Item</button>;
    }
    ```

#### Pagination and Infinite Scrolling
- **Theory**: Pagination involves splitting data into discrete pages, while infinite scrolling loads more data as the user scrolls to the bottom of the page.
- **Significance**: Both techniques improve performance and user experience by reducing the initial load time and allowing users to consume data incrementally.
- **Usage**: Implement pagination by fetching only a subset of data or implement infinite scrolling by triggering data fetch on scroll events.
  - **Example** (Pagination with Axios):
    ```javascript
    const [page, setPage] = useState(1);
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(false);

    const fetchData = async () => {
      setLoading(true);
      const response = await axios.get(`https://api.example.com/data?page=${page}`);
      setData(prevData => [...prevData, ...response.data]);
      setLoading(false);
    };

    useEffect(() => {
      fetchData();
    }, [page]);

    const loadMore = () => setPage(prevPage => prevPage + 1);

    return (
      <div>
        <ul>
          {data.map(item => (
            <li key={item.id}>{item.name}</li>
          ))}
        </ul>
        {loading ? <p>Loading...</p> : <button onClick={loadMore}>Load More</button>}
      </div>
    );
    ```
  - **Example** (Infinite Scrolling with react-query):
    ```javascript
    import { useInfiniteQuery } from 'react-query';
    import axios from 'axios';

    const fetchData = async ({ pageParam = 1 }) => {
      const { data } = await axios.get(`https://api.example.com/data?page=${pageParam}`);
      return data;
    };

    function Component() {
      const { data, fetchNextPage, hasNextPage, isFetching, isFetchingNextPage } = useInfiniteQuery(
        'fetchData',
        fetchData,
        {
          getNextPageParam: (lastPage, pages) => lastPage.nextPage ?? false,
        }
      );

      return (
        <div>
          <ul>
            {data?.pages.map((page, index) => (
              <React.Fragment key={index}>
                {page.items.map(item => (
                  <li key={item.id}>{item.name}</li>
                ))}
              </React.Fragment>
            ))}
          </ul>
          <button
            onClick={() => fetchNextPage()}
            disabled={!hasNextPage || isFetchingNextPage}
          >
            {isFetchingNextPage ? 'Loading more...' : hasNextPage ? 'Load More' : 'Nothing more to load'}
          </button>
          <div>{isFetching && !isFetchingNextPage ? 'Fetching...' : null}</div>
        </div>
      );
    }
    ```

This comprehensive course content for Module 6 covers essential concepts of API integration in React applications, including fetching data, handling loading and error states, making HTTP requests using Axios, and advanced data-fetching strategies using libraries like SWR and react-query. Each section offers theoretical insights, practical examples, and an explanation of their significance to ensure a robust understanding and application of these concepts in real-world projects.