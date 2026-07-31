import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen } from "lucide-react";
import { Panel } from "./ui";
import guideSource from "../../docs/08_USER_GUIDE.md?raw";

// Renders docs/08_USER_GUIDE.md in-app, so the guide has one source of truth
// on disk and one rendered view here, rather than two copies to keep in sync.

export default function UserGuide() {
  return (
    <Panel title="User Guide" subtitle="Every screen, field, and term explained" icon={<BookOpen size={16} />}>
      <article className="guide-prose max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{guideSource}</ReactMarkdown>
      </article>
    </Panel>
  );
}
