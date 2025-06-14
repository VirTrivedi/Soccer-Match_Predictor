import React from 'react';

const Header: React.FC = () => {
  return (
    <header style={{ 
      backgroundColor: '#004085', /* Darker blue */
      padding: '15px 30px', 
      color: 'white',
      textAlign: 'left', /* Align text to left */
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)' /* Subtle shadow */
    }}>
      <h2 style={{ margin: 0, fontSize: '1.5em', fontWeight: '500' }}>Soccer Predictor Dashboard</h2>
    </header>
  );
};

export default Header;