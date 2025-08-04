# Two-Column Chat UI Test Plan

## Test Scenarios

### 1. Layout Verification
- **Test**: Open the page in a browser
- **Expected**: Two-column layout with chat on left, panels on right
- **Verify**: 
  - Chat interface maintains all existing functionality
  - Right column shows two collapsed panels
  - Layout is responsive on different screen sizes

### 2. Panel Toggle Functionality
- **Test**: Click each panel header to expand/collapse
- **Expected**: 
  - Panels smoothly expand and collapse
  - Chevron icons rotate appropriately
  - ARIA attributes update correctly (aria-expanded)
  - Keyboard navigation works (Enter/Space keys)

### 3. Prebuilt Citations Panel
- **Test**: Send a chat message that triggers citations
- **Expected**:
  - Citations appear in both the original sources area and the prebuilt citations panel
  - Panel auto-expands when new citations are received
  - Citations display with proper formatting (source, score, text)

### 4. Dynamic KG Updates Panel
- **Test**: Observe the dynamic KG panel during conversation
- **Expected**:
  - Mock triples appear periodically (every 7 seconds with 30% chance)
  - Panel auto-expands when new triples are added
  - Triples show subject → predicate → object format
  - Confidence scores and timestamps are displayed

### 5. Responsive Design
- **Test**: Resize browser window to different breakpoints
- **Expected**:
  - Desktop (>1200px): Full two-column layout
  - Tablet (768px-1200px): Narrower right column
  - Mobile (<768px): Single column with panels above chat
  - All content remains accessible and functional

### 6. Accessibility
- **Test**: Navigate using keyboard and screen reader
- **Expected**:
  - All panels are keyboard accessible
  - ARIA attributes provide proper context
  - Focus indicators are visible
  - Content is semantically structured

### 7. Performance
- **Test**: Monitor performance during heavy citation updates
- **Expected**:
  - Debounced updates prevent UI flooding
  - Smooth animations without lag
  - Memory usage remains stable

## Mock Data Testing

The implementation includes mock data generators for testing:
- Prebuilt citations simulate every 5 seconds (20% chance)
- Dynamic triples simulate every 7 seconds (30% chance)

## Integration Points

To integrate with actual graph managers:
1. Replace `initializeMockGraphEvents()` with real event listeners
2. Connect `updatePrebuiltCitations()` to actual citation API
3. Connect `updateDynamicKg()` to dynamic graph update events
4. Remove mock simulation intervals

## Browser Compatibility

Tested features:
- CSS Grid support
- CSS Transitions
- ES6+ JavaScript features
- WebSocket connections
- Local Storage