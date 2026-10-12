import { Send, Square } from "lucide-react";
import { Link } from "react-router";
import type { ProfileSummary, ProviderConfig } from "../../lib/api";
import { Select, TextArea } from "../Input";

// Value of the synthetic option that mirrors a provider which is not one of
// the saved profiles: an inline config, or a profile deleted since it was
// picked. Choosing it is a no-op so the current config is kept as-is.
const INLINE_VALUE = "__inline__";

const HINT_TITLE =
  "Sets the active provider for the whole workspace; a run already in progress keeps its model";

// Composer at the foot of the live view: a follow-up objective, or feedback
// on a finished swarm build. The model picker lives here because a follow-up
// is posted with the workspace provider, and the only other places to change
// that are the Settings and Overview pages -- the Agent page itself used to
// show the model only as a mismatch warning while a run was in flight.
export function FollowUpBar({
  reply, setReply, onSend, pending, running, onCancel, revisable,
  profiles, provider, onPickProfile,
}: {
  reply: string; setReply: (s: string) => void; onSend: () => void;
  pending: boolean; running?: boolean;
  onCancel?: () => void | Promise<void>;
  // The session holds a finished swarm build: the message is feedback that
  // revises it (SWARM_REVISE -> SWARM_VERIFY) rather than a new objective.
  revisable?: boolean;
  // Saved LLM profiles offered for the next message, the workspace provider
  // the pick lands in, and the store action that applies a profile to it.
  profiles: ProfileSummary[];
  provider: ProviderConfig;
  onPickProfile: (p: ProfileSummary) => void;
}) {
  return (
    <div className="px-4 pt-2.5 pb-3 border-t border-muted bg-slate-950/60 flex items-end gap-2">
      <TextArea
        rows={2}
        value={reply}
        onChange={(e) => setReply(e.target.value)}
        placeholder={revisable
          ? "Describe what to change (e.g. “make the top bar span the full width”, “remember the API key after a reload”)…"
          : "Post a follow-up objective…"}
        disabled={pending}
        className="flex-1"
        aria-label={revisable ? "Feedback on the build" : "Follow-up objective"}
      />
      {running && onCancel && (
        <button
          type="button" onClick={() => { void onCancel(); }}
          title="Stop after the current step"
          className="av-btn-ghost text-red-300 hover:text-red-200"
        ><Square className="h-3 w-3" /> Stop</button>
      )}
      <ModelPicker profiles={profiles} provider={provider} onPick={onPickProfile} />
      <button
        type="button" onClick={onSend}
        disabled={pending || !reply.trim()}
        className="av-btn-primary"
        title={revisable ? "Apply this feedback to the build and re-verify" : "Send"}
      ><Send className="h-3 w-3" /> {revisable ? "Revise" : "Send"}</button>
    </div>
  );
}

// Picks the saved profile the next message is sent with. Applying a profile
// sets the workspace-wide provider (exactly what Settings does), hence the
// tooltip; a drain already in progress keeps the deps it was dispatched with
// (``_schedule_drain`` coalesces onto it), so the pick only affects what is
// sent next.
function ModelPicker({ profiles, provider, onPick }: {
  profiles: ProfileSummary[];
  provider: ProviderConfig;
  onPick: (p: ProfileSummary) => void;
}) {
  const hint = (
    <p className="truncate text-[10px] font-mono text-slate-500" title={HINT_TITLE}>
      {`next message uses ${provider.model}`}
    </p>
  );
  if (profiles.length === 0) {
    return (
      <div className="shrink-0 max-w-44 pb-0.5">
        {hint}
        <Link to="/settings" className="text-[10px] font-mono text-emerald-400 hover:underline">
          add a profile
        </Link>
      </div>
    );
  }
  // A controlled <select> whose value matches no option shows the first one
  // as selected, which would claim a profile the provider isn't using; so a
  // provider that is not one of the saved profiles gets a synthetic entry
  // instead, named after the stale profile when there is one.
  const current = provider.use_profile
    ? profiles.find((p) => p.name === provider.profile_name) : undefined;
  const selected = current ? current.name : INLINE_VALUE;
  const synthetic = provider.use_profile && provider.profile_name
    ? provider.profile_name : "Inline";
  return (
    <div className="shrink-0 w-44 flex flex-col gap-1">
      <Select
        aria-label="Model for the next message"
        value={selected}
        onChange={(e) => {
          const picked = profiles.find((p) => p.name === e.target.value);
          if (picked) onPick(picked);
        }}
        className="py-1.5"
      >
        {selected === INLINE_VALUE && (
          <option value={INLINE_VALUE}>{`${synthetic} · ${provider.model}`}</option>
        )}
        {profiles.map((p) => (
          <option key={p.name} value={p.name}>{`${p.name} · ${p.model}`}</option>
        ))}
      </Select>
      {hint}
    </div>
  );
}
