Key changes made to fix the blank charts:

1. **Separate Figures for Each Chart**:
   - Created three independent Plotly figures (`fig_size`, `fig_lines`, `fig_tokens`)
   - Each figure has its own layout and configuration
   - Converted each figure to JSON separately

2. **Simplified JavaScript Initialization**:
   - Directly initialize each chart with its own figure data
   - Removed complex trace extraction logic
   - Simplified the update function to directly set x-axis ranges

3. **Improved Date Range Handling**:
   - Consolidated date range calculation from all available data
   - Fixed the date range picker initialization
   - Ensured all charts update simultaneously with the same date range

4. **Error Handling**:
   - Added null checks for missing data
   - Improved fallback behavior when specific metrics are unavailable
   - Enhanced logging for troubleshooting

The solution maintains all existing functionality (sound effects, pipeline status, etc.) while ensuring the charts render correctly with proper interactivity and date synchronization.