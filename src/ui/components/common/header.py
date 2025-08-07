"""
Application Header Component

Provides the main header with styling and branding for the QCM Generator Pro interface.
"""

import streamlit as st
import os


class ApplicationHeader:
    """Main application header component."""
    
    def __init__(self, title: str = "QCM Generator Pro", subtitle: str = "Génération automatique de QCM multilingues à partir de documents PDF"):
        """
        Initialize header component.
        
        Args:
            title: Main application title
            subtitle: Application subtitle/description
        """
        self.title = title
        self.subtitle = subtitle
        # Force disable debug mode
        self._disable_debug_mode()
    
    def render_css(self) -> None:
        """Render custom CSS styles for the application."""
        st.markdown("""
        <style>
        /* Hide Streamlit debug/development elements */
        .streamlit-debug-tree-wrapper,
        .streamlit-debug-tree,
        .streamlit-debug,
        .stElementToolbar,
        .stElementToolbarButton,
        .stAlert[data-baseweb="alert"],
        .stDeployButton,
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        .css-18ni7ap.e8zbici2,
        .css-1dp5vir.e8zbici1,
        header[data-testid="stHeader"],
        .stAppViewContainer > .main > div > div > div > section.main > div:first-child {
            display: none !important;
        }
        
        /* More aggressive hiding of sidebar debug elements and page navigation */
        [data-testid="stSidebar"] > div:first-child > div:first-child,
        [data-testid="stSidebar"] .element-container:first-of-type,
        [data-testid="stSidebar"] .streamlit-expanderHeader:first-of-type,
        [data-testid="stSidebar"] .streamlit-expander:first-of-type,
        [data-testid="stSidebar"] div[data-stale="false"]:first-child,
        [data-testid="stSidebar"] .css-1cypcdb.eczjsme11,
        .css-1cypcdb.eczjsme11,
        /* Hide automatic page navigation */
        [data-testid="stSidebar"] .stSelectbox:first-of-type,
        [data-testid="stSidebar"] .stSelectbox[data-baseweb="select"]:first-of-type,
        [data-testid="stSidebar"] > div > div:first-child > div > div > div,
        [data-testid="stSidebar"] .css-1d391kg,
        .css-1d391kg,
        /* Hide Streamlit's automatic page selector */
        div[data-testid="stSidebar"] .stSelectbox[aria-label*="page"],
        div[data-testid="stSidebar"] .stSelectbox[aria-label*="Page"],
        div[data-testid="stSidebar"] > div:first-child > div > div:first-child,
        /* Target the page navigation container */
        .css-1rs6os.edgvbvh3,
        .css-17ziqus.edgvbvh1,
        .css-1rs6os,
        .edgvbvh3,
        .edgvbvh1 {
            display: none !important;
        }
        
        /* Hide any element that looks like debug info */
        div[style*="background-color: rgb(240, 242, 246)"],
        div[style*="border: 1px solid rgb(255, 75, 75)"],
        div[style*="color: rgb(255, 75, 75)"],
        .streamlit-container .element-container:first-child,
        .streamlit-container .stSelectbox:first-child {
            display: none !important;
        }
        
        /* Application styles */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .status-success {
            padding: 1rem;
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .status-error {
            padding: 1rem;
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .status-info {
            padding: 1rem;
            background-color: #cce5ff;
            border-left: 4px solid #0066cc;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        /* Clean up sidebar styling */
        [data-testid="stSidebar"] {
            padding-top: 0rem !important;
        }
        </style>
        
        <script>
        // JavaScript to hide Streamlit's automatic page navigation
        function hidePageNavigation() {
            // Hide any selectbox or navigation elements in sidebar
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                const firstElement = sidebar.querySelector('div:first-child > div:first-child');
                if (firstElement) {
                    firstElement.style.display = 'none';
                }
                
                // Hide any selectbox that might be a page navigator
                const selectboxes = sidebar.querySelectorAll('.stSelectbox');
                selectboxes.forEach((selectbox, index) => {
                    if (index === 0) { // Hide first selectbox which is likely the page navigator
                        selectbox.style.display = 'none';
                    }
                });
            }
        }
        
        // Run immediately and then periodically to catch dynamic elements
        hidePageNavigation();
        setInterval(hidePageNavigation, 500);
        
        // Also run on DOM changes
        if (typeof MutationObserver !== 'undefined') {
            const observer = new MutationObserver(hidePageNavigation);
            observer.observe(document.body, { childList: true, subtree: true });
        }
        </script>
        """, unsafe_allow_html=True)
    
    def render_header(self) -> None:
        """Render the main application header."""
        st.markdown(f"""
        <div class="main-header">
            <h1>🎯 {self.title}</h1>
            <p>{self.subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _disable_debug_mode(self) -> None:
        """Force disable Streamlit debug mode."""
        # Set environment variables to disable debug features
        os.environ["STREAMLIT_CLIENT_SHOWERRORDTAILS"] = "false"
        os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"
        os.environ["STREAMLIT_CLIENT_TOOLBARMODE"] = "minimal"
        
        # Try to disable any existing debug configurations
        try:
            if hasattr(st, '_config'):
                st._config.set_option('client.showErrorDetails', False)
                st._config.set_option('client.toolbarMode', 'minimal')
        except:
            pass  # Ignore if config is not accessible
    
    def render(self) -> None:
        """Render both CSS and header."""
        self.render_css()
        self.render_header()