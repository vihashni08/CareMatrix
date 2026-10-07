/**
 * Dedicated SSE Service for CareMatrix.
 * Manages EventSource connection, handles reconnection states, parses SSE envelopes,
 * and passes normalized payloads into the central store dispatch.
 */

window.CareMatrixSseService = function(dispatch) {
  let eventSource = null;
  let reconnectTimer = null;
  let isIntentionallyClosed = false;

  const connect = function() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }

    dispatch({ type: 'SET_SSE_STATUS', payload: 'RECONNECTING' });

    try {
      eventSource = new EventSource('/api/events/stream');

      eventSource.onopen = function() {
        dispatch({ type: 'SET_SSE_STATUS', payload: 'CONNECTED' });
      };

      eventSource.onerror = function() {
        dispatch({ type: 'SET_SSE_STATUS', payload: 'RECONNECTING' });
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
        if (!isIntentionallyClosed && !reconnectTimer) {
          reconnectTimer = setTimeout(function() {
            reconnectTimer = null;
            connect();
          }, 3000);
        }
      };

      eventSource.onmessage = function(rawEvent) {
        if (!rawEvent || !rawEvent.data) return;
        try {
          const envelope = JSON.parse(rawEvent.data);
          if (!envelope || !envelope.type) return;

          // Dispatch the live event directly into the state reducer
          dispatch({
            type: 'SSE_EVENT_RECEIVED',
            payload: {
              type: envelope.type,
              data: envelope.data,
              timestamp: envelope.timestamp || Date.now() / 1000,
            }
          });
        } catch (err) {
          console.warn('[CareMatrix SSE] Malformed message ignored:', err, rawEvent.data);
        }
      };
    } catch (e) {
      console.error('[CareMatrix SSE] Failed to initialize EventSource:', e);
      dispatch({ type: 'SET_SSE_STATUS', payload: 'DISCONNECTED' });
    }
  };

  const disconnect = function() {
    isIntentionallyClosed = true;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    dispatch({ type: 'SET_SSE_STATUS', payload: 'DISCONNECTED' });
  };

  return {
    connect: connect,
    disconnect: disconnect,
  };
};

