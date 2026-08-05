import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router";
import AppShell from "./layout/AppShell";

// Each route is code-split so the initial bundle only includes the shell.
// Markdown/highlight.js live in AskPage + PlanPage and are pulled lazily.
const OverviewPage = lazy(() => import("./pages/OverviewPage"));
const AskPage = lazy(() => import("./pages/AskPage"));
const PlanPage = lazy(() => import("./pages/PlanPage"));
const AgentPage = lazy(() => import("./pages/AgentPage"));
const IndexPage = lazy(() => import("./pages/IndexPage"));
const ActivityPage = lazy(() => import("./pages/ActivityPage"));
const AdminPage = lazy(() => import("./pages/AdminPage"));
const SettingsPage = lazy(() => import("./pages/SettingsPage"));

function RouteFallback() {
  return (
    <div className="h-full w-full flex items-center justify-center text-xs text-slate-500 font-mono">
      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse mr-2" />
      loading…
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route
          path="/"
          element={
            <Suspense fallback={<RouteFallback />}>
              <OverviewPage />
            </Suspense>
          }
        />
        <Route
          path="/ask"
          element={
            <Suspense fallback={<RouteFallback />}>
              <AskPage />
            </Suspense>
          }
        />
        <Route
          path="/plan"
          element={
            <Suspense fallback={<RouteFallback />}>
              <PlanPage />
            </Suspense>
          }
        />
        <Route
          path="/agent"
          element={
            <Suspense fallback={<RouteFallback />}>
              <AgentPage />
            </Suspense>
          }
        />
        <Route
          path="/index"
          element={
            <Suspense fallback={<RouteFallback />}>
              <IndexPage />
            </Suspense>
          }
        />
        <Route
          path="/activity"
          element={
            <Suspense fallback={<RouteFallback />}>
              <ActivityPage />
            </Suspense>
          }
        />
        <Route
          path="/admin"
          element={
            <Suspense fallback={<RouteFallback />}>
              <AdminPage />
            </Suspense>
          }
        />
        <Route
          path="/settings"
          element={
            <Suspense fallback={<RouteFallback />}>
              <SettingsPage />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
