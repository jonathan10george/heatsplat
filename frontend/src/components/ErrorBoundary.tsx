import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  label: string;
}

interface State {
  error: Error | null;
}

/** Keeps a crash in one panel (e.g. WebGL) from blanking the whole app. */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`${this.props.label} crashed`, error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="stage-error" role="alert">
          {this.props.label} failed: {this.state.error.message}
        </div>
      );
    }
    return this.props.children;
  }
}
