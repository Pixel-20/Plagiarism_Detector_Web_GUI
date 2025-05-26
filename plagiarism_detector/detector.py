import os
import sys
import re
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import shutil
import numpy as np
import networkx as nx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plagiarism_detector.log'),
        logging.StreamHandler()
    ]
)

class Node:
    """Simple tree node class for ZSS algorithm."""
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children else []
        self._size = None

class TokenProcessor:
    """Process and normalize code tokens."""
    
    # Define token significance levels
    TOKEN_SIGNIFICANCE = {
        'KEYWORD': 1.0,      # High significance
        'IDENTIFIER': 0.8,
        'LITERAL': 0.7,
        'PUNCTUATION': 0.2,   # Low significance
        'COMMENT': 0.1       # Very low significance
    }
    
    # Tokens to filter out completely
    IGNORED_TOKENS = {';', '{', '}', '(', ')', ','}
    
    @classmethod
    def filter_tokens(cls, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
        """Filter and weight tokens based on their significance."""
        filtered_tokens = []
        
        for token_kind, token_text in tokens:
            # Skip ignored tokens
            if token_text in cls.IGNORED_TOKENS:
                continue
                
            # Get token significance
            significance = cls.TOKEN_SIGNIFICANCE.get(token_kind, 0.5)
            
            # Add to filtered list with significance weight
            filtered_tokens.append((token_kind, token_text, significance))
        
        return filtered_tokens
    
    @staticmethod
    def normalize_identifiers(tokens: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """Normalize identifiers while preserving their role."""
        id_map = {}
        normalized_tokens = []
        
        for token_kind, token_text, significance in tokens:
            if token_kind == 'IDENTIFIER':
                if token_text not in id_map:
                    id_map[token_text] = f'ID_{len(id_map)}'
                normalized_tokens.append((token_kind, id_map[token_text], significance))
            else:
                normalized_tokens.append((token_kind, token_text, significance))
        
        return normalized_tokens

    @staticmethod
    def extract_cpp_patterns(tokens):
        """Extract C++ specific patterns from tokens."""
        patterns = {
            'memory_management': [],
            'template_usage': [],
            'stl_usage': [],
            'pointer_usage': []
        }
        
        # Look for memory management patterns
        for i, (kind, text, _) in enumerate(tokens):
            if text in ['new', 'delete', 'malloc', 'free', 'alloc']:
                patterns['memory_management'].append((i, text))
            elif text in ['template', 'typename', 'class'] and kind == 'KEYWORD':
                patterns['template_usage'].append((i, text))
            elif text in ['vector', 'map', 'set', 'list', 'queue', 'stack', 'array', 'deque']:
                patterns['stl_usage'].append((i, text))
            elif text in ['*', '&', '->', 'nullptr', 'NULL']:
                patterns['pointer_usage'].append((i, text))
        
        return patterns

class ASTAnalyzer:
    """Simplified AST analyzer that works without clang bindings."""
    
    def __init__(self, file_path: str):
        import re  # Import re module at the top level of the class
        self.re = re  # Store as instance variable for use in other methods
        self.file_path = file_path
        self.ast_graph = nx.DiGraph()
        self.subtree_hashes = {}
        self.cfg_graph = nx.DiGraph()  # Control Flow Graph
        self.tokens = []
        
        try:
            # Since we can't use clang bindings in a web app easily,
            # we'll use a simpler token-based approach
            self._parse_file()
        except Exception as e:
            logging.error(f"Failed to parse {file_path}: {str(e)}")
            raise
    
    def _parse_file(self):
        """Parse file into tokens using a simple approach."""
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Simple tokenization (this is a simplified approach)
            # re module is already imported in __init__
            
            # Keywords in C/C++
            keywords = {
                'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double',
                'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int', 'long', 'register',
                'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef',
                'union', 'unsigned', 'void', 'volatile', 'while', 'class', 'namespace', 'try',
                'catch', 'new', 'delete', 'public', 'private', 'protected', 'template', 'this',
                'virtual', 'friend', 'inline', 'operator', 'throw', 'bool', 'true', 'false'
            }
            
            # Tokenize
            token_pattern = r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\w+|[^\s\w]'
            tokens = []
            
            for match in self.re.finditer(token_pattern, content, self.re.MULTILINE | self.re.DOTALL):
                token_text = match.group(0)
                
                # Determine token kind
                if token_text.startswith('//') or token_text.startswith('/*'):
                    token_kind = 'COMMENT'
                elif token_text.startswith('"') or token_text.startswith("'"):
                    token_kind = 'LITERAL'
                elif token_text in keywords:
                    token_kind = 'KEYWORD'
                elif self.re.match(r'^\w+$', token_text):
                    token_kind = 'IDENTIFIER'
                else:
                    token_kind = 'PUNCTUATION'
                
                tokens.append((token_kind, token_text))
            
            self.tokens = tokens
            
            # Build a simple AST representation
            self._build_simple_ast(content)
            
        except Exception as e:
            logging.error(f"Error parsing file {self.file_path}: {str(e)}")
            raise
    
    def _build_simple_ast(self, content):
        """Build a simplified AST representation."""
        # This is a very simplified approach
        # In a real implementation, we would use a proper parser
        
        # Create root node
        root_id = "root"
        self.ast_graph.add_node(root_id, kind="ROOT", spelling="root", depth=0)
        
        # Track braces to estimate nesting
        brace_stack = []
        current_depth = 0
        current_parent = root_id
        node_counter = 0
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for function or class declarations
            if self.re.search(r'(class|struct|enum)\s+\w+', line):
                node_id = f"class_{node_counter}"
                node_counter += 1
                self.ast_graph.add_node(node_id, kind="CLASS_DECL", spelling=line, depth=current_depth+1)
                self.ast_graph.add_edge(current_parent, node_id)
                
                if '{' in line:
                    brace_stack.append((node_id, current_depth+1))
                    current_parent = node_id
                    current_depth += 1
            
            elif self.re.search(r'\w+\s+\w+\s*\([^)]*\)\s*({|$)', line):
                node_id = f"function_{node_counter}"
                node_counter += 1
                self.ast_graph.add_node(node_id, kind="FUNCTION_DECL", spelling=line, depth=current_depth+1)
                self.ast_graph.add_edge(current_parent, node_id)
                
                if '{' in line:
                    brace_stack.append((node_id, current_depth+1))
                    current_parent = node_id
                    current_depth += 1
            
            # Check for control structures
            elif self.re.search(r'^\s*(if|for|while|switch)\s*\(', line):
                node_id = f"control_{node_counter}"
                node_counter += 1
                self.ast_graph.add_node(node_id, kind="CONTROL_STMT", spelling=line, depth=current_depth+1)
                self.ast_graph.add_edge(current_parent, node_id)
                
                if '{' in line:
                    brace_stack.append((node_id, current_depth+1))
                    current_parent = node_id
                    current_depth += 1
            
            # Track braces for nesting
            elif '{' in line and not brace_stack:
                node_id = f"block_{node_counter}"
                node_counter += 1
                self.ast_graph.add_node(node_id, kind="COMPOUND_STMT", spelling="{", depth=current_depth+1)
                self.ast_graph.add_edge(current_parent, node_id)
                brace_stack.append((node_id, current_depth+1))
                current_parent = node_id
                current_depth += 1
            
            # Handle closing braces
            if '}' in line and brace_stack:
                current_parent, current_depth = brace_stack.pop()
        
        # Calculate subtree hashes
        for node in self.ast_graph.nodes():
            self.subtree_hashes[node] = self._hash_subtree(node)
    
    def _hash_subtree(self, node):
        """Compute a hash for the subtree rooted at node."""
        # Get node data
        data = self.ast_graph.nodes[node]
        
        # Use generic labels for consistent hashing
        node_str = self._get_node_generic_label(data)
        
        # Get children
        children = list(self.ast_graph.successors(node))
        
        # Sort children by their generic labels to ensure consistent ordering
        # This helps detect similar structures even if child order differs
        children_with_labels = [(child, self._get_node_generic_label(self.ast_graph.nodes[child])) for child in children]
        sorted_children = [child for child, _ in sorted(children_with_labels, key=lambda x: x[1])]
        
        # Get hashes for sorted children
        child_hashes = [self._hash_subtree(child) for child in sorted_children]
        
        # Combine node and child hashes
        combined = node_str + '_'.join(map(str, sorted(child_hashes)))
        return hash(combined)
    
    def _get_node_generic_label(self, node_data: Dict) -> str:
        """Generate a generic label for AST nodes that's resilient to minor code changes."""
        kind = node_data.get('kind', 'UNKNOWN')
        spelling = node_data.get('spelling', '')

        if kind == 'CONTROL_STMT':
            # Extract the control keyword (for, while, if, etc.)
            control_keyword = spelling.split('(')[0].strip().split()[-1] if '(' in spelling else spelling.strip().split()[-1]
            
            # Normalize all loop constructs to a generic LOOP label
            if control_keyword in ['for', 'while', 'do', 'foreach']: 
                return f"LOOP_CONSTRUCT"
                
            # Normalize all conditional constructs
            if control_keyword in ['if', 'else', 'switch', 'case', 'ternary']:
                return f"CONDITIONAL_CONSTRUCT"
                
            # Default case
            return f"{kind}_GENERIC"
            
        elif kind == 'FUNCTION_DECL':
            # For function declarations, just use the generic type
            # This makes it more resilient to function name changes
            return f"FUNCTION_CONSTRUCT"
            
        elif kind == 'CLASS_DECL':
            # For class declarations, just use the generic type
            return f"CLASS_CONSTRUCT"
            
        elif kind == 'COMPOUND_STMT' or spelling == '{' or spelling == '}':
            # Generic label for code blocks
            return f"BLOCK_CONSTRUCT"
            
        # For expressions and operators, generalize based on type
        elif 'expr' in kind.lower() or kind in ['BINARY_OPERATOR', 'UNARY_OPERATOR']:
            return f"EXPRESSION_CONSTRUCT"
            
        # For variable declarations, use a generic label
        elif 'decl' in kind.lower() or 'var' in kind.lower():
            return f"VARIABLE_CONSTRUCT"
            
        # For other kinds, if spelling is too specific or not useful, just use kind
        return f"{kind}_GENERIC"

    def _to_zss_tree(self, node) -> Node:
        """Convert AST node to format suitable for tree comparison."""
        data = self.ast_graph.nodes[node]
        # Use generic label
        label = self._get_node_generic_label(data)
        
        children = [self._to_zss_tree(child) for child in self.ast_graph.successors(node)]
        return Node(label, children)
    
    def analyze(self) -> Dict:
        """Perform complete AST analysis."""
        try:
            # Get root node
            root = next(iter(self.ast_graph.nodes())) if self.ast_graph.nodes() else None
            if not root:
                raise ValueError("Empty AST graph")
            
            # Create ZSS tree
            zss_tree = self._to_zss_tree(root)
            
            # Calculate structural metrics
            depths = [d['depth'] for _, d in self.ast_graph.nodes(data=True) if 'depth' in d]
            max_depth = max(depths) if depths else 0
            node_types = {d.get('kind', 'UNKNOWN') for _, d in self.ast_graph.nodes(data=True)}
            
            # Count loops and control structures
            loops = sum(1 for _, d in self.ast_graph.nodes(data=True) 
                      if d.get('kind') == 'CONTROL_STMT' and 
                      ('for' in d.get('spelling', '') or 'while' in d.get('spelling', '')))
            
            # Estimate complexity
            time_complexity = "O(n)" if loops > 0 else "O(1)"
            if loops > 1:
                time_complexity = f"O(n^{min(loops, 3)})"
            
            return {
                'ast_depth': max_depth,
                'node_count': len(self.ast_graph),
                'node_types': list(node_types),
                'subtree_hashes': self.subtree_hashes,
                'zss_tree': zss_tree,
                'graph_structure': {
                    'edges': len(self.ast_graph.edges()),
                    'avg_branching': len(self.ast_graph.edges()) / max(1, len(self.ast_graph.nodes())),
                    'leaf_nodes': sum(1 for n in self.ast_graph.nodes() 
                                    if self.ast_graph.out_degree(n) == 0)
                },
                'complexity': {
                    'time_complexity': time_complexity,
                    'loop_count': loops,
                    'max_nesting': max_depth
                }
            }
            
        except Exception as e:
            logging.error(f"Error in AST analysis for {self.file_path}: {str(e)}")
            raise

class ScalableSimilarityAnalyzer:
    """Analyze code similarity without using heavy ML libraries."""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.submissions = {}
    
    def add_submission(self, file_id: str, tokens: List[str]):
        """Add a submission for comparison."""
        self.submissions[file_id] = tokens
    
    @staticmethod
    def compute_tree_distance(tree1: Node, tree2: Node) -> float:
        """Compute normalized tree edit distance with optimized matching."""
        def tree_size(node):
            if node._size is not None:
                return node._size
            size = 1 + sum(tree_size(child) for child in node.children)
            node._size = size
            return size
        
        # Pre-compute sizes
        size1 = tree_size(tree1)
        size2 = tree_size(tree2)
        
        # If sizes are extremely different, we can short-circuit
        # But use a more lenient threshold (0.9 instead of 0.8)
        if abs(size1 - size2) / max(size1, size2) > 0.9:
            return 0.0
            
        # Extract node type from label (first part before underscore)
        def get_node_type(label):
            # For our new consistent labels, we can just use the whole label
            if any(label.startswith(prefix) for prefix in ["LOOP_", "CONDITIONAL_", "FUNCTION_", "CLASS_", "BLOCK_", "EXPRESSION_", "VARIABLE_"]):
                return label
            return label.split('_')[0] if '_' in label else label
            
        # Count node types in trees (for structural similarity)
        def count_node_types(node, counts=None):
            if counts is None:
                counts = {}
            
            node_type = get_node_type(node.label)
            counts[node_type] = counts.get(node_type, 0) + 1
            
            for child in node.children:
                count_node_types(child, counts)
            
            return counts
            
        # Get node type distributions
        types1 = count_node_types(tree1)
        types2 = count_node_types(tree2)
        
        # Calculate node type distribution similarity
        all_types = set(types1.keys()) | set(types2.keys())
        type_sim = sum(min(types1.get(t, 0), types2.get(t, 0)) for t in all_types) / \
                  sum(max(types1.get(t, 0), types2.get(t, 0)) for t in all_types) if all_types else 0
        
        # Simple tree distance calculation with memoization
        def simple_distance(t1, t2, memo=None):
            if memo is None:
                memo = {}
            
            key = (id(t1), id(t2))
            if key in memo:
                return memo[key]
            
            # If one tree is empty, return size of other tree
            if not t1.children and not t2.children:
                # For leaf nodes, exact match is best but partial match is possible
                # if labels share the same prefix (e.g., CONTROL_STMT_*)
                if t1.label == t2.label:
                    result = 0  # Perfect match
                elif get_node_type(t1.label) == get_node_type(t2.label):
                    result = 0.2  # Same type, different details - reduced penalty further
                else:
                    result = 0.8  # Different types - reduced penalty
                memo[key] = result
                return result
            
            # Calculate cost of replacing root
            # More lenient matching - if labels share the same prefix, partial match
            if t1.label == t2.label:
                replace_cost = 0  # Perfect match
            elif get_node_type(t1.label) == get_node_type(t2.label):
                replace_cost = 0.2  # Same type, different details - reduced penalty
            else:
                replace_cost = 0.8  # Different types - reduced penalty
            
            # Calculate cost of matching children
            if not t1.children:
                result = replace_cost + len(t2.children) * 0.5  # Further reduced penalty
                memo[key] = result
                return result
            
            if not t2.children:
                result = replace_cost + len(t1.children) * 0.5  # Further reduced penalty
                memo[key] = result
                return result
            
            # Try to match children - improved algorithm to find best matches
            # This is an approximation of the optimal assignment problem
            costs = []
            for c1 in t1.children:
                c1_costs = []
                for c2 in t2.children:
                    cost = simple_distance(c1, c2, memo)
                    c1_costs.append(cost)
                costs.append(min(c1_costs) if c1_costs else 0)
            
            child_match_cost = sum(costs)
            
            # Penalize different number of children, but less severely
            child_count_penalty = abs(len(t1.children) - len(t2.children)) * 0.3  # Further reduced penalty
            
            result = replace_cost + child_match_cost + child_count_penalty
            memo[key] = result
            return result
        
        # Compute distance
        distance = simple_distance(tree1, tree2)
        max_size = max(size1, size2)
        
        # Normalize and invert (0 = completely different, 1 = identical)
        # More lenient normalization and incorporate type distribution similarity
        edit_sim = 1 - (distance / (max_size * 2.0))  # Even more lenient normalization
        
        # Combine edit distance with type distribution similarity
        combined_sim = (edit_sim * 0.7) + (type_sim * 0.3)
        
        # Clamp to [0, 1]
        return max(0, min(1, combined_sim))
    
    @staticmethod
    def compute_cfg_similarity(cfg1: nx.DiGraph, cfg2: nx.DiGraph) -> float:
        """Compute similarity between control flow graphs."""
        if not cfg1.nodes() or not cfg2.nodes():
            return 0.0
            
        # Compare basic graph properties
        nodes_sim = min(len(cfg1), len(cfg2)) / max(len(cfg1), len(cfg2))
        edges_sim = min(len(cfg1.edges()), len(cfg2.edges())) / max(len(cfg1.edges()), len(cfg2.edges())) if max(len(cfg1.edges()), len(cfg2.edges())) > 0 else 1.0
        
        # Compare node types distribution
        types1 = {}
        types2 = {}
        
        for _, data in cfg1.nodes(data=True):
            node_type = data.get('type', 'unknown')
            types1[node_type] = types1.get(node_type, 0) + 1
            
        for _, data in cfg2.nodes(data=True):
            node_type = data.get('type', 'unknown')
            types2[node_type] = types2.get(node_type, 0) + 1
            
        # Jaccard similarity of node types
        all_types = set(types1.keys()) | set(types2.keys())
        type_sim = sum(min(types1.get(t, 0), types2.get(t, 0)) for t in all_types) / sum(max(types1.get(t, 0), types2.get(t, 0)) for t in all_types) if all_types else 0
        
        # Weighted combination
        return 0.3 * nodes_sim + 0.3 * edges_sim + 0.4 * type_sim
    
    @staticmethod
    def compute_complexity_similarity(comp1: Dict, comp2: Dict) -> float:
        """Compare algorithmic complexity signatures."""
        # Time complexity comparison
        time_match = comp1['time_complexity'] == comp2['time_complexity']
        
        # Loop count similarity
        loop_sim = 1 - abs(comp1['loop_count'] - comp2['loop_count']) / max(comp1['loop_count'], comp2['loop_count']) if max(comp1['loop_count'], comp2['loop_count']) > 0 else 1.0
        
        # Nesting depth similarity
        nesting_sim = 1 - abs(comp1['max_nesting'] - comp2['max_nesting']) / max(comp1['max_nesting'], comp2['max_nesting']) if max(comp1['max_nesting'], comp2['max_nesting']) > 0 else 1.0
        
        # Weighted combination
        return 0.5 * (1 if time_match else 0) + 0.25 * loop_sim + 0.25 * nesting_sim
    
    @staticmethod
    def compute_weighted_similarity(submission1: Dict, submission2: Dict) -> Dict:
        """Compute comprehensive similarity with weighted metrics."""
        try:
            # Direct token comparison (most reliable for exact matches)
            tokens1 = [t[1] for t in submission1['token_features']['normalized']]
            tokens2 = [t[1] for t in submission2['token_features']['normalized']]
            
            # Calculate token-based similarity
            common_tokens = set(tokens1) & set(tokens2)
            all_tokens = set(tokens1) | set(tokens2)
            token_sim = len(common_tokens) / len(all_tokens) if all_tokens else 0
            
            # Calculate sequence similarity (for exact matches)
            min_len = min(len(tokens1), len(tokens2))
            max_len = max(len(tokens1), len(tokens2))
            
            # Check for sequence matches
            matches = 0
            for i in range(min_len):
                if tokens1[i] == tokens2[i]:
                    matches += 1
            
            sequence_sim = matches / max_len if max_len > 0 else 0
            
            # Tree edit distance similarity
            tree_sim = ScalableSimilarityAnalyzer.compute_tree_distance(
                submission1['ast_features']['zss_tree'],
                submission2['ast_features']['zss_tree']
            )
            
            # Subtree hash similarity
            hashes1 = set(submission1['ast_features']['subtree_hashes'].values())
            hashes2 = set(submission2['ast_features']['subtree_hashes'].values())
            hash_sim = len(hashes1.intersection(hashes2)) / len(hashes1.union(hashes2)) if hashes1 or hashes2 else 0
            
            # Enhanced subtree similarity - consider partial matches
            # Get the most frequent subtree patterns
            hash_freq1 = {}
            for h in submission1['ast_features']['subtree_hashes'].values():
                hash_freq1[h] = hash_freq1.get(h, 0) + 1
                
            hash_freq2 = {}
            for h in submission2['ast_features']['subtree_hashes'].values():
                hash_freq2[h] = hash_freq2.get(h, 0) + 1
            
            # Get top patterns by frequency
            top_patterns1 = sorted(hash_freq1.items(), key=lambda x: x[1], reverse=True)[:min(20, len(hash_freq1))]
            top_patterns2 = sorted(hash_freq2.items(), key=lambda x: x[1], reverse=True)[:min(20, len(hash_freq2))]
            
            # Calculate similarity of top patterns
            top_hashes1 = {h for h, _ in top_patterns1}
            top_hashes2 = {h for h, _ in top_patterns2}
            
            # Enhanced similarity calculation
            common_top = len(top_hashes1.intersection(top_hashes2))
            total_top = len(top_hashes1.union(top_hashes2))
            
            # Weight the top patterns more heavily
            enhanced_hash_sim = (hash_sim * 0.4) + (common_top / total_top * 0.6) if total_top > 0 else hash_sim
            
            # Structure similarity
            struct1 = submission1['ast_features']['graph_structure']
            struct2 = submission2['ast_features']['graph_structure']
            struct_sim = 1 - abs(struct1['avg_branching'] - struct2['avg_branching']) / \
                         max(struct1['avg_branching'], struct2['avg_branching']) if max(struct1['avg_branching'], struct2['avg_branching']) > 0 else 1.0
            
            # Complexity similarity
            complexity_sim = ScalableSimilarityAnalyzer.compute_complexity_similarity(
                submission1['ast_features'].get('complexity', {'time_complexity': 'O(1)', 'loop_count': 0, 'max_nesting': 0}),
                submission2['ast_features'].get('complexity', {'time_complexity': 'O(1)', 'loop_count': 0, 'max_nesting': 0})
            )
            
            # CFG similarity
            # Assuming cfg_graph is available in ast_features or submission directly
            # If ASTAnalyzer.analyze doesn't provide this, this will default to 0 or raise an error if keys are missing.
            cfg1 = submission1['ast_features'].get('cfg_graph', nx.DiGraph()) # Default to empty graph
            cfg2 = submission2['ast_features'].get('cfg_graph', nx.DiGraph()) # Default to empty graph
            if not isinstance(cfg1, nx.DiGraph): cfg1 = nx.DiGraph() # Ensure it's a graph
            if not isinstance(cfg2, nx.DiGraph): cfg2 = nx.DiGraph() # Ensure it's a graph
            
            cfg_sim = ScalableSimilarityAnalyzer.compute_cfg_similarity(cfg1, cfg2)
            
            # Weighted combination with optimized weights
            weights = {
                'token': 0.10,       # Reduced - less important for logical plagiarism
                'sequence': 0.02,    # Further reduced - too sensitive to restructuring
                'tree': 0.45,        # Further increased - most important for logical similarity
                'hash': 0.25,        # Increased - now more effective with enhanced calculation
                'cfg': 0.10,         # Reduced - less reliable than tree similarity
                'structure': 0.05,   # Maintained - useful supplementary metric
                'complexity': 0.03   # Reduced - less reliable than structural metrics
            }
            
            # Calculate overall similarity using weighted average
            metrics = [token_sim, sequence_sim, tree_sim, enhanced_hash_sim, cfg_sim, struct_sim, complexity_sim]
            metric_weights = [weights['token'], weights['sequence'], weights['tree'], weights['hash'], 
                             weights['cfg'], weights['structure'], weights['complexity']]
            
            # Calculate weighted average
            overall_sim = sum(m * w for m, w in zip(metrics, metric_weights))
            
            # Also track the maximum similarity for reference
            max_sim = max(metrics) if metrics else 0.0
            
            # Flag suspicious similarities
            flags = []
            if max_sim >= 0.95:
                flags.append("HIGHLY_SUSPICIOUS")
            elif max_sim >= 0.85:
                flags.append("SUSPICIOUS")
                
            # If any structural metric is high but token similarity is low, flag as potential logical plagiarism
            if (tree_sim >= 0.75 or enhanced_hash_sim >= 0.75) and token_sim < 0.50:
                flags.append("POTENTIAL_LOGICAL_PLAGIARISM")
                
            # If tree similarity is extremely high, flag regardless of token similarity
            if tree_sim >= 0.90:
                flags.append("STRUCTURAL_MATCH")
                
            # Calculate confidence based on agreement of metrics
            # Focus on structural metrics for logical plagiarism
            structural_metrics = [tree_sim, enhanced_hash_sim, cfg_sim]
            structural_avg = sum(structural_metrics) / len(structural_metrics)
            
            # Higher confidence if structural metrics agree with each other
            structural_variance = sum((m - structural_avg) ** 2 for m in structural_metrics) / len(structural_metrics)
            confidence = 1 - (structural_variance * 2)  # Lower variance means higher confidence
            confidence = max(0, min(1, confidence))  # Clamp to [0,1]
            
            return {
                'token_similarity': token_sim,
                'sequence_similarity': sequence_sim,
                'tree_edit_similarity': tree_sim,
                'subtree_hash_similarity': enhanced_hash_sim,
                'cfg_similarity': cfg_sim,
                'structure_similarity': struct_sim,
                'complexity_similarity': complexity_sim,
                'overall_similarity': overall_sim,
                'max_similarity': max_sim,
                'flags': flags,
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error in similarity calculation: {str(e)}")
            return {
                'overall_similarity': 0.0,
                'tree_edit_similarity': 0.0,
                'subtree_hash_similarity': 0.0,
                'structure_similarity': 0.0,
                'confidence': 0.0
            }

class EnhancedPlagiarismDetector:
    """Main plagiarism detector class."""
    
    def __init__(self, output_dir: str = 'plagiarism_results', 
                 similarity_threshold: float = 0.8,
                 max_workers: int = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.max_workers = max_workers or os.cpu_count()
        
        self.submissions = {}
        self.comparison_results = []
        self.lsh_analyzer = ScalableSimilarityAnalyzer(threshold=similarity_threshold)
        
        # Set up logging for this instance
        self.logger = logging.getLogger(f"PlagiarismDetector_{id(self)}")
    
    def _analyze_single_file(self, file_path: str) -> Optional[Dict]:
        """Analyze a single file with comprehensive error handling."""
        try:
            self.logger.info(f"Analyzing {file_path}")
            
            # Parse and analyze AST
            ast_analyzer = ASTAnalyzer(file_path)
            ast_features = ast_analyzer.analyze()
            
            # Process tokens
            tokens = ast_analyzer.tokens
            filtered_tokens = TokenProcessor.filter_tokens(tokens)
            normalized_tokens = TokenProcessor.normalize_identifiers(filtered_tokens)
            
            # Extract C++ specific patterns
            cpp_patterns = TokenProcessor.extract_cpp_patterns(normalized_tokens)
            
            results = {
                'file_path': file_path,
                'ast_features': ast_features,
                'token_features': {
                    'original': tokens,
                    'filtered': filtered_tokens,
                    'normalized': normalized_tokens,
                    'cpp_patterns': cpp_patterns
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Add to submissions
            token_strings = [t[1] for t in normalized_tokens]
            self.lsh_analyzer.add_submission(file_path, token_strings)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}", exc_info=True)
            return None
    
    def analyze_files(self, file_paths: List[str]):
        """Analyze multiple files in parallel."""
        self.logger.info(f"Starting analysis of {len(file_paths)} files")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self._analyze_single_file, fp): fp 
                             for fp in file_paths}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        self.submissions[file_path] = result
                except Exception as e:
                    self.logger.error(f"Analysis failed for {file_path}: {str(e)}")
    
    def find_similarities(self):
        """Find similar submissions."""
        self.logger.info("Starting similarity analysis")
        self.comparison_results = []
        
        # Compare all pairs of submissions
        file_paths = list(self.submissions.keys())
        
        for i, file1 in enumerate(file_paths):
            for file2 in file_paths[i+1:]:
                # Calculate similarity
                similarity = self.lsh_analyzer.compute_weighted_similarity(
                    self.submissions[file1],
                    self.submissions[file2]
                )
                
                # Determine if this pair exceeds threshold using either overall or max similarity
                overall_sim = similarity['overall_similarity']
                max_sim = similarity['max_similarity']
                tree_sim = similarity['tree_edit_similarity']
                
                # Use a smarter approach to determine effective similarity
                # - For logical plagiarism, tree similarity is more important
                # - For direct copies, overall similarity is more important
                # - Max similarity provides a good upper bound
                
                # Weight tree similarity higher for logical plagiarism detection
                if tree_sim >= 0.85 and similarity.get('token_similarity', 0) < 0.5:
                    # Likely logical plagiarism case
                    effective_sim = (tree_sim * 0.7) + (max_sim * 0.3)
                else:
                    # General case - use a blend of overall and max similarity
                    effective_sim = (overall_sim * 0.6) + (max_sim * 0.4)
                
                # Add flags for borderline cases
                if 'flags' not in similarity:
                    similarity['flags'] = []
                    
                if effective_sim >= self.similarity_threshold * 0.9 and effective_sim < self.similarity_threshold:
                    similarity['flags'].append("BORDERLINE_SIMILARITY")
                
                if effective_sim >= self.similarity_threshold or "STRUCTURAL_MATCH" in similarity.get('flags', []):
                    self.comparison_results.append({
                        'file1': file1,
                        'file2': file2,
                        'similarity_metrics': similarity,
                        'overall_similarity': overall_sim,
                        'max_similarity': max_sim,
                        'effective_similarity': effective_sim,
                        'flags': similarity.get('flags', []),
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Sort results by effective similarity (descending)
        self.comparison_results.sort(key=lambda x: x['effective_similarity'], reverse=True)
        
        return self.comparison_results 