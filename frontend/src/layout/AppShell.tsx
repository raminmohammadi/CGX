import { useEffect } from "react";
import { Outlet } from "react-router-dom";
import Header from "./Header";
import Sidebar from "./Sidebar";
import ConnectionBanner from "../components/ConnectionBanner";
import { startConnectionPoller } from "../store/connection";
import { useTrace } from "../store/trace";

export default function AppShell() {
  // Single global poller owned by the shell; banner + header subscribe.
  useEffect(() => startConnectionPoller(), []);
  // Prime the trace flag once on shell mount so the Header pill reflects
  // the server-side state (incl. CGX_TRACE env pinning) on first paint.
  useEffect(() => {
    void useTrace.getState().refresh();
  }, []);

  return (
    <div className="h-full flex flex-col">
      <Header />
      <ConnectionBanner />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar />
        <main className="flex-1 bg-base relative overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
