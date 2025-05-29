"""Main application entry point for the JSON RAG System."""

import logging
import gradio as gr
from core_system import JSONRAGSystem
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def initialize_rag_system():
    """Initialize the RAG system."""
    logger.info("Starting JSON RAG System...")
    rag_system = JSONRAGSystem()
    
    # Initialize the system
    success, message = rag_system.initialize_system()
    if not success:
        logger.error(f"Failed to initialize RAG system: {message}")
        return None, f"Initialization failed: {message}"
    
    return rag_system, "System initialized successfully!"

def process_query(query, history, session_id=None):
    """Process user query and return response."""
    if not query.strip():
        return history, "Please enter a valid question.", session_id
    
    try:
        # Initialize RAG system if not already done
        if not hasattr(process_query, 'rag_system') or process_query.rag_system is None:
            process_query.rag_system, init_message = initialize_rag_system()
            if process_query.rag_system is None:
                return history + [(query, f"Error: {init_message}")], "", session_id
        
        # Process the query
        response, new_session_id, search_results = process_query.rag_system.process_query(query, session_id)
        
        # Format the response for display
        formatted_response = format_response(response, search_results)
        
        # Update conversation history
        history.append((query, formatted_response))
        
        return history, "", new_session_id
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_response = f"Sorry, I encountered an error: {str(e)}"
        history.append((query, error_response))
        return history, "", session_id

def format_response(response, search_results):
    """Format the response for better display."""
    formatted = response
    
    if search_results:
        formatted += "\n\n**Search Results:**\n"
        for i, result in enumerate(search_results, 1):
            source_doc = result.get('source_document', {})
            score = result.get('combined_score', 0)
            
            # Format each result nicely
            formatted += f"\n**Result {i}** (Score: {score:.2f})\n"
            
            # Show key fields from the source document
            key_fields = ['name', 'property_type', 'price', 'accommodates', 'review_scores_rating']
            for field in key_fields:
                if field in source_doc:
                    formatted += f"- **{field.replace('_', ' ').title()}**: {source_doc[field]}\n"
            
            # Show a snippet of the full JSON
            formatted += f"\n```json\n{json.dumps(source_doc, indent=2)[:500]}...\n```\n"
    
    return formatted

def get_system_status():
    """Get current system status."""
    if hasattr(process_query, 'rag_system') and process_query.rag_system:
        try:
            status = process_query.rag_system.get_system_status()
            status_text = f"""
**System Status**
- Documents Indexed: {status.get('documents_indexed', 0)}
- Queries Processed: {status.get('query_count', 0)}
- Active Sessions: {status.get('active_sessions', 0)}
- Average Response Time: {status.get('average_response_time', 0):.2f}s
- System Initialized: {'✅' if status.get('system_initialized', False) else '❌'}
            """
            return status_text
        except Exception as e:
            return f"Error getting status: {e}"
    else:
        return "System not initialized"

def clear_history():
    """Clear conversation history."""
    return [], "", None

def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="JSON RAG System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 JSON RAG System
            
            An intelligent search and question-answering system for JSON documents.
            Ask questions about the data and get comprehensive answers with source information.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=500,
                    label="Conversation",
                    show_label=True,
                    container=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ask a question",
                        placeholder="e.g., 'Find apartments under $200 with good ratings'",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear History", variant="secondary")
                    example_btn = gr.Button("Show Examples", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### System Information")
                
                status_display = gr.Markdown(
                    value=get_system_status(),
                    label="System Status"
                )
                
                refresh_btn = gr.Button("Refresh Status", variant="secondary")
                
                gr.Markdown(
                    """
                    ### Example Queries
                    
                    - "Find 2-bedroom apartments with WiFi"
                    - "Show luxury properties with pools"
                    - "What are budget-friendly options under $100?"
                    - "Find highly rated properties by superhosts"
                    - "Show me places that accommodate 6+ people"
                    """
                )
        
        # Hidden state for session management
        session_state = gr.State(value=None)
        
        # Event handlers
        def respond(message, history, session_id):
            return process_query(message, history, session_id)
        
        # Submit button and enter key
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, session_state],
            outputs=[chatbot, msg, session_state]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot, session_state],
            outputs=[chatbot, msg, session_state]
        )
        
        # Clear history
        clear_btn.click(
            clear_history,
            outputs=[chatbot, msg, session_state]
        )
        
        # Refresh status
        refresh_btn.click(
            get_system_status,
            outputs=[status_display]
        )
        
        # Example button
        def show_examples():
            examples = [
                "Find 2-bedroom apartments with WiFi and good ratings",
                "Show me luxury properties with pools",
                "What are budget-friendly options under $100?",
                "Find highly rated properties by superhosts",
                "Show places that accommodate 6+ people"
            ]
            return gr.update(choices=examples, visible=True)
        
    return demo

def main():
    """Main function to run the application."""
    logger.info("Starting JSON RAG System...")
    
    # Create the interface
    demo = create_interface()
    
    logger.info("Launching web interface...")
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()
