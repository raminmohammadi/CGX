import { useEffect } from "react";
import { Outlet } from "react-router-dom";
import Header from "./Header";
import Sidebar from "./Sidebar";
import ConnectionBanner from "../components/ConnectionBanner";
import { startConnectionPoller } from "../store/connection";

export default function AppShell() {
  // Single global poller owned by the shell; banner + header subscribe.
  useEffect(() => startConnectionPoller(), []);

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
