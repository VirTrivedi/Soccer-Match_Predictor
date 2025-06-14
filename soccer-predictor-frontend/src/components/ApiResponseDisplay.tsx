import React from 'react';

interface ApiResponseDisplayProps {
  data: any;
  title?: string;
}

const ApiResponseDisplay: React.FC<ApiResponseDisplayProps> = ({ data, title = "API Response" }) => {
  if (!data) return null;

  return (
    <div style={{ textAlign: 'left', backgroundColor: '#f9f9f9', border: '1px solid #eee', padding: '10px', marginTop: '10px', maxHeight: '300px', overflowY: 'auto' }}>
      <h4>{title}</h4>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
};

export default ApiResponseDisplay;