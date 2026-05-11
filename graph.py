import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExecutionGraph:
    """Manages explicit dependency relationships between events."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        logger.info(f"ExecutionGraph initialized with DB: {db_path}")

    def _get_connection(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_edge(self, from_event_id: str, to_event_id: str, 
                 edge_type: str, session_id: str):
        """Adds an edge to the graph."""
        if edge_type not in ['tool_input', 'tool_output', 'decision', 'follows']:
            logger.warning(f"Invalid edge type: {edge_type}")
            return
        
        try:
            conn = self._get_connection()
            conn.execute(
                """INSERT OR IGNORE INTO execution_graph 
                   (from_event_id, to_event_id, edge_type, session_id, created_at)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (from_event_id, to_event_id, edge_type, session_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")

    def get_neighbors(self, event_id: str, depth: int = 1, visited: Optional[set] = None) -> List[Dict[str, Any]]:
        """
        Traverses the graph bidirectionally.
        Returns list of neighbor event IDs up to depth.
        Includes cycle detection (via visited set).
        """
        if visited is None:
            visited = set()
        
        if depth <= 0 or event_id in visited:
            return []
        
        visited.add(event_id)
        neighbors = []
        
        try:
            conn = self._get_connection()
            # Bidirectional: find edges where event_id is 'from' or 'to'
            cursor = conn.execute(
                """SELECT from_event_id, to_event_id, edge_type, session_id 
                   FROM execution_graph 
                   WHERE from_event_id = ? OR to_event_id = ?""",
                (event_id, event_id)
            )
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                neighbor_id = row['to_event_id'] if row['from_event_id'] == event_id else row['from_event_id']
                if neighbor_id not in visited:
                    neighbors.append({
                        "event_id": neighbor_id,
                        "edge_type": row['edge_type'],
                        "session_id": row['session_id']
                    })
                    # Recurse
                    if depth > 1:
                        deeper = self.get_neighbors(neighbor_id, depth-1, visited)
                        neighbors.extend(deeper)
            return neighbors
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return []

    def get_edge_types(self, event_id: str) -> List[str]:
        """Returns list of edge types connected to this event."""
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                """SELECT DISTINCT edge_type FROM execution_graph 
                   WHERE from_event_id = ? OR to_event_id = ?""",
                (event_id, event_id)
            )
            types = [row['edge_type'] for row in cursor.fetchall()]
            conn.close()
            return types
        except Exception as e:
            logger.error(f"Failed to get edge types: {e}")
            return []
