function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export class ReplayEngine {
  constructor({ onUpdate }) {
    this.onUpdate = onUpdate;
    this.events = [];
    this.durationMs = 0;
    this.positionMs = 0;
    this.speed = 1;
    this.isPlaying = false;
    this.tickMs = 20;
    this.intervalId = null;
  }

  setTimeline(events) {
    this.pause();
    this.events = Array.isArray(events) ? events : [];
    const last = this.events[this.events.length - 1];
    this.durationMs = last ? last.end_ms : 0;
    this.positionMs = 0;
    this.emit();
  }

  setSpeed(speed) {
    const next = Number(speed);
    if (!Number.isFinite(next) || next <= 0) {
      return;
    }
    this.speed = next;
    this.emit();
  }

  play() {
    if (this.isPlaying || this.durationMs === 0) {
      return;
    }

    this.isPlaying = true;
    this.intervalId = setInterval(() => {
      const delta = this.tickMs * this.speed;
      this.seek(this.positionMs + delta);
      if (this.positionMs >= this.durationMs) {
        this.pause();
      }
    }, this.tickMs);
    this.emit();
  }

  pause() {
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    if (this.isPlaying) {
      this.isPlaying = false;
      this.emit();
    }
  }

  reset() {
    this.pause();
    this.seek(0);
  }

  stepForward() {
    this.pause();
    const boundaries = this.events
      .map((event) => event.end_ms)
      .filter((value, index, all) => {
        if (index === 0) return true;
        return Math.abs(value - all[index - 1]) > 1e-5;
      });

    const nextBoundary = boundaries.find((value) => value > this.positionMs + 1e-5);
    if (nextBoundary === undefined) {
      this.seek(this.durationMs);
      return;
    }

    this.seek(nextBoundary);
  }

  stepBackward() {
    this.pause();
    const boundaries = [0, ...this.events.map((event) => event.start_ms)];
    const previous = boundaries
      .filter((value) => value < this.positionMs - 1e-5)
      .sort((a, b) => b - a)[0];

    this.seek(previous === undefined ? 0 : previous);
  }

  seek(ms) {
    this.positionMs = clamp(ms, 0, this.durationMs);
    this.emit();
  }

  getSnapshot() {
    const activeIndex = this.events.findIndex((event) => {
      if (event.duration_ms <= 0) {
        return this.positionMs === event.start_ms;
      }
      return this.positionMs >= event.start_ms && this.positionMs < event.end_ms;
    });

    const finalIndex = this.events.length - 1;
    const resolvedActiveIndex =
      activeIndex >= 0
        ? activeIndex
        : this.positionMs >= this.durationMs && finalIndex >= 0
          ? finalIndex
          : 0;

    const activeEvent = this.events[resolvedActiveIndex] || null;
    const stageElapsedMs = activeEvent
      ? Math.max(0, this.positionMs - activeEvent.start_ms)
      : 0;
    const stageProgress =
      activeEvent && activeEvent.duration_ms > 0
        ? clamp(stageElapsedMs / activeEvent.duration_ms, 0, 1)
        : 1;

    return {
      isPlaying: this.isPlaying,
      speed: this.speed,
      positionMs: this.positionMs,
      durationMs: this.durationMs,
      progress: this.durationMs > 0 ? this.positionMs / this.durationMs : 0,
      activeIndex: resolvedActiveIndex,
      activeEvent,
      stageProgress,
    };
  }

  emit() {
    if (typeof this.onUpdate === "function") {
      this.onUpdate(this.getSnapshot());
    }
  }
}
