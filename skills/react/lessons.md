# Lessons learned for skill `react`

Auto-recorded by CGX after failed/repaired runs. Each is a durable rule fed back into future generation. Safe to prune: delete a line here and its matching entry in the `.jsonl` beside it.

- Ensure all necessary dependencies are installed and correctly configured before running tests.  <!-- id=lsn_99ad81268a6e 2026-10-11 -->
- Verify that environment variables are properly set for all required configurations.  <!-- id=lsn_f521bf3a9778 2026-10-11 -->
- Implement comprehensive test coverage to catch potential issues early in the development cycle.  <!-- id=lsn_a9a777f661d9 2026-10-11 -->
- Ensure that all components and modules are correctly imported and used in the application.  <!-- id=lsn_55ed1e5a79cd 2026-10-11 -->
- Ensure that the import statements in React components match the actual exports in the imported modules.  <!-- id=lsn_83810027b7e2 2026-10-11 -->
- Avoid using nested React Router components to prevent runtime errors and ensure proper routing.  <!-- id=lsn_6fa17268def9 2026-10-11 -->
- Always declare and import dependencies in React components to prevent runtime ReferenceErrors.  <!-- id=lsn_2c3eb146b8f9 2026-10-11 -->
- A React component should not render its own <Routes> if the route table is already defined in a parent component to prevent duplicate rendering and navigation state issues.  <!-- id=lsn_81b4ccfa5852 2026-10-11 -->
- Ensure that all destructured props in React components are correctly passed from their parent components.  <!-- id=lsn_4209a045e1d9 2026-10-11 -->
- React components should not redeclare routes that are already managed by a parent component to prevent double rendering and navigation state issues.  <!-- id=lsn_10784e4cef66 2026-10-11 -->
- Ensure that imported modules match their export types; if a module exports named functions, import them accordingly to avoid runtime errors.  <!-- id=lsn_bc13eb927b8e 2026-10-11 -->
- React components should properly utilize props passed from their parent components to maintain data flow and functionality.  <!-- id=lsn_9a2e3456de6c 2026-10-11 -->
- React components should consistently import named exports from a module, avoiding default imports when the module only provides named exports.  <!-- id=lsn_0c989e2b6732 2026-10-11 -->
- Ensure that all React components handle potential errors during rendering to prevent the entire app from crashing.  <!-- id=lsn_5e78941b2f75 2026-10-11 -->
- When using React, verify that all imported components and modules are correctly configured and that their exports match the imports.  <!-- id=lsn_db2b4cc47bf3 2026-10-11 -->
- Ensure all named exports are correctly imported in React components.  <!-- id=lsn_95d74e0629f6 2026-10-11 -->
- Wrap the entire React app with a <BrowserRouter> or <RouterProvider> to enable routing.  <!-- id=lsn_a9f183ea7d6f 2026-10-11 -->
- Verify that components accept all necessary props passed by their callers.  <!-- id=lsn_9f125d9f0a3a 2026-10-11 -->
- Define all global variables and dependencies at the top level of your application to prevent runtime errors.  <!-- id=lsn_87c373c3984f 2026-10-11 -->
- React components should import named exports correctly from their respective modules.  <!-- id=lsn_49b51e2329e5 2026-10-11 -->
- React components should utilize all props passed to them from their parent components.  <!-- id=lsn_382f6790727b 2026-10-11 -->
- Ensure that imported modules use the correct named exports. For example, if a module exports named functions like `load`, `save`, and `clearSettings`, the import statement should specify these names explicitly.  <!-- id=lsn_b4bca6ce6c5e 2026-10-11 -->
- Avoid nested React Router components to prevent runtime errors. Ensure that only one instance of `BrowserRouter` or `HashRouter` is used in the application.  <!-- id=lsn_c5751299b4c2 2026-10-11 -->
- Verify that components accept all necessary props passed by their parents. Ensure that destructuring in component props includes all expected properties.  <!-- id=lsn_475dcc2ab538 2026-10-11 -->
- Handle potential null or undefined data before using array methods like `map` to prevent runtime errors. Ensure that data is properly initialized and validated before rendering.  <!-- id=lsn_370ed2ee8f99 2026-10-11 -->
